"""Sweep scheduling, honest scope, command remapping and real1/1 validation; no device timing."""
import contextlib
import copy
import io
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import score_bitmap_scales as s


def scheduling_and_commands():
    assert s.SIZES==(1024,8192,65536,262144,1048576,4194304)
    s.validate_sizes(list(s.SIZES))
    for sizes in ([],[8192,1024],[1024,1024],[10000000],[1000]):
        try:s.validate_sizes(sizes)
        except ValueError:pass
        else:raise AssertionError('Unsupported sweep accepted')
    for n in s.SIZES:
        seen={}
        for i in range(6):
            order=s.process_order(n,i);assert set(order)=={'baseline','candidate',*s.cpu_roles(n)}
            seen['candidate_first']=seen.get('candidate_first',0)+(order.index('candidate')<order.index('baseline'))
        assert seen['candidate_first']==3
        assert len(s.cpu_roles(n))==(4 if n<=65536 else 2)
    prior={'command':['npu','old_index','oldA','oldB','old_out','5','oldvendor','10','8','24','5','30','old_refA','old_refB']}
    before=copy.deepcopy(prior);changed=s.command_for(prior,Path('new_input'),Path('new_output'),Path('candidate'))
    assert prior==before and changed[5]=='5' and changed[7:12]==['10','8','24','5','30']
    assert changed[1:5]==[str(Path('new_input/index')),str(Path('new_input/A.txt')),str(Path('new_input/B.txt')),'new_output']
    assert changed[12:]==[str(Path('new_input/A/expected_scores.bin')),str(Path('new_input/B/expected_scores.bin'))]
    cpu={'command':['cpu','index','A','B','out','32','10','32','8','24','5','30']}
    before=copy.deepcopy(cpu);single=s.command_for(cpu,Path('input'),Path('out'),single=True)
    assert cpu==before and single[5:]==['1','10','1','8','24','5','30']
    assert s.command_for(cpu,Path('input'),Path('out'))[5:]==before['command'][5:]


def decisions_and_curve():
    def item(ms):return {'summary':{c:{'median_batch_ms':ms,'p95_batch_ms':ms} for c in ('A','B')}}
    rows=[{'baseline':item(1),'candidate':item(.9),'cpu':item(2),'cpu_directory':item(1.8),
        'cpu_single':item(.2),'cpu_directory_single':item(.3)} for _ in range(6)]
    small=s.point_decision(rows,1024)
    assert small['stable_at_least_5pct_gain_over_original_npu']
    assert small['versus_original_frozen32_pair']['performance_target_met_at_this_scale']
    assert not small['versus_all_declared_cpu_configs']['performance_target_met_at_this_scale']
    medium=s.point_decision(rows,4194304);assert medium['versus_all_declared_cpu_configs']['performance_target_met_at_this_scale']
    for n in (1024,4194304):
        common=s.cpu_gate(rows,('cpu','cpu_directory'));actual=s.point_decision(rows,n)['versus_original_frozen32_pair']
        assert {k:v for k,v in common.items() if k!='cpu_roles'}==actual
    reused={'docs':10000000,'decision':{'versus_frozen_cpus':{'performance_target_met_at_this_scale':True}}}
    points=[{'docs':1024,'status':'six_rounds_checked','decision':small,'cpu_roles':list(s.cpu_roles(1024))},
        {'docs':4194304,'status':'six_rounds_checked','decision':medium,'cpu_roles':list(s.cpu_roles(4194304))}]
    curve=s.curve_summary(points,reused)
    assert curve['fresh_coverage_complete'] and curve['first_observed_winning_docs_against_common_frozen32_pair']==1024
    assert curve['first_observed_winning_docs_under_declared_cpu_policy']==4194304
    assert curve['declared_cpu_policy_points_with_session_origin'][-1]['session']=='reused_previous_10m'
    points[1]['status']='not_measured_device_budget';curve=s.curve_summary(points,reused)
    assert not curve['fresh_coverage_complete'] and curve['first_observed_winning_docs_under_declared_cpu_policy']==10000000


def single_cpu_validator():
    # Temporary synthetic session-contract fixtures only. No fake result is
    # persisted, used as hardware evidence or delivered to a benchmark driver.
    with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as folder:
        folder=Path(folder);inputs=folder/'input'
        cases={'A':[[0,0,1]]*8,'B':[[2,3,2]]*8}
        with contextlib.redirect_stdout(io.StringIO()):s.corpus.prepare(inputs,1024,8,950,cases)
        desc,index=s.corpus.validate(inputs)
        samples=[{'cycle':i,'case':'B' if j^(i&1) else 'A','call_wall_ms':.2,'internal_search_ms':.1,
            'query_h2d_bytes':0,'topk_d2h_bytes':0} for i in range(30) for j in range(2)]
        for role in ('cpu_single','cpu_directory_single'):
            out=folder/role;out.mkdir()
            for cp in s.g.checks.CHECKPOINTS:
                target=out/cp;target.mkdir();gold=inputs/cp[-1]
                for dst,src in (('scores.bin','expected_scores.bin'),('top_ids.bin','expected_top_ids.bin'),('top_scores.bin','expected_top_scores.bin')):
                    shutil.copyfile(gold/src,target/dst)
            data={'status':'search_benchmark_calls_completed_checks_pending','backend':'cpu','docs':1024,
                'queries':8,'k':10,'partitions':1,'requested_threads':1,'actual_threads':1,'capacity_queries':8,'capacity_words':24,
                'input_contract':'raw_grouped_INT32_word_IDs_in_CPU_memory','timer_policy':s.g.checks.TIMER,'index_load_batches':1,
                'index_bytes':index['index_bytes'],'index_readback_byte_identical':True,'warmup_cycles':5,'warmup_search_calls':10,
                'measured_cycles':30,'measured_search_calls':60,'measured_topk_bit_checks':60,'checkpoint_search_calls':4,
                'measured_file_io_calls':0,'measured_debug_download_calls':0,'measured_poison_calls':0,
                'diagnostic_checkpoints_outside_measurement':True,'fp32_flags':'-O3 -march=native -ffp-contract=off -fno-fast-math',
                'host_source_sha256':s.g.checks.sources(s.g.checks.CPU_SOURCES if role=='cpu_single' else s.g.HOST_SOURCES),'samples':samples}
            path=out/'benchmark_session.json';s.r.write_json(path,data);original=path.read_bytes()
            result=s.verify_single_cpu(out,inputs,desc,index,role)
            assert result['session']['actual_threads']==1 and path.read_bytes()==original
            for field,value in (('actual_threads',32),('partitions',32),('measured_file_io_calls',1),('timer_policy','kernel_only')):
                bad=copy.deepcopy(data);bad[field]=value;s.r.write_json(path,bad)
                try:s.verify_single_cpu(out,inputs,desc,index,role)
                except ValueError:pass
                else:raise AssertionError('Weakened single-CPU contract accepted: '+field)
            s.r.write_json(path,data);file=out/'before_A/scores.bin';raw=bytearray(file.read_bytes());raw[0]^=1;file.write_bytes(raw)
            try:s.verify_single_cpu(out,inputs,desc,index,role)
            except ValueError:pass
            else:raise AssertionError('Wrong single-thread full scores accepted')


if __name__=='__main__':
    scheduling_and_commands();decisions_and_curve();single_cpu_validator()
    print('PASS: six size schedules, balanced paired order, unchanged command capacities, real1/1 CPU contracts, full-score corruption rejection, separate CPU/gain gates and historical10m origin. No CANN/device/performance evidence.')
