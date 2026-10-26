"""Complete small/medium bitmap coverage using installed operators; reuse accepted10m evidence."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import score_bitmap as bitmap
x=bitmap.retained;g=x.g;r=x.r;corpus=x.corpus;digest=corpus.digest
SIZES=(1024,8192,65536,262144,1048576,4194304)
OWN=('standalone/score_bitmap_scales.py',)+bitmap.OWN+x.OWN+x.HOST
CHECKED='actual_bitmap_small_medium_scale_coverage_checked'
INCOMPLETE='bitmap_scale_coverage_incomplete_not_all_points_measured'


def cpu_roles(n):
    # Preserve the original32/32 opponents. Add the SAME binaries at1/1 only
    # at small scales; no CPU implementation/build change or thread sweep.
    return ('cpu','cpu_directory','cpu_single','cpu_directory_single') if n<=65536 else ('cpu','cpu_directory')


def process_order(n,round_index):
    order=('cpu','baseline','candidate','cpu_directory')
    if n<=65536:order+=('cpu_single','cpu_directory_single')
    return order if round_index%2==0 else tuple(reversed(order))


def validate_sizes(sizes):
    r.require(sizes==sorted(set(sizes)) and sizes and all(type(n) is int and 1024<=n<10000000 for n in sizes),
        'Unique increasing small/medium article counts required; completed10m is reused, not rerun')


def cpu_gate(rows,roles):
    r.require(len(rows)==6,'Six complete rounds required');cases={}
    for case in ('A','B'):
        cpu=[min(row[role]['summary'][case]['median_batch_ms'] for role in roles) for row in rows]
        npu=[row['candidate']['summary'][case]['median_batch_ms'] for row in rows]
        cp95=[min(row[role]['summary'][case]['p95_batch_ms'] for role in roles) for row in rows]
        np95=[row['candidate']['summary'][case]['p95_batch_ms'] for row in rows]
        wins=sum(a<b for a,b in zip(npu,cpu))
        cases[case]={'cpu_medians_ms':cpu,'npu_medians_ms':npu,'cpu_p95_ms':cp95,'npu_p95_ms':np95,
            'wins':wins,'cpu_over_npu':statistics.median(cpu)/statistics.median(npu),
            'passed':wins==6 and statistics.median(np95)<statistics.median(cp95)}
    return {'performance_target_met_at_this_scale':all(v['passed'] for v in cases.values()),'cpu_roles':list(roles),'cases':cases}


def point_decision(rows,n):
    paired=bitmap.decision(rows)
    return {'versus_verified_npu':paired['versus_verified_npu'],
        'stable_at_least_5pct_gain_over_original_npu':all(v['passed'] for v in paired['versus_verified_npu'].values()),
        'versus_original_frozen32_pair':paired['versus_frozen_cpus'],
        'versus_all_declared_cpu_configs':cpu_gate(rows,cpu_roles(n)),
        'scope':'Small-scale CPU loss or sub5% gain is a measured boundary, not a correctness failure and does not stop the sweep.'}


def curve_summary(points,reused):
    measured=[p for p in points if p.get('status')=='six_rounds_checked']
    fresh_wins=[p['docs'] for p in measured if p['decision']['versus_all_declared_cpu_configs']['performance_target_met_at_this_scale']]
    primary=[{'docs':p['docs'],'passed':p['decision']['versus_original_frozen32_pair']['performance_target_met_at_this_scale'],'session':'fresh_small_medium'} for p in measured]
    primary.append({'docs':reused['docs'],'passed':reused['decision']['versus_frozen_cpus']['performance_target_met_at_this_scale'],'session':'reused_previous_10m'})
    primary.sort(key=lambda p:p['docs']);wins=[p['docs'] for p in primary if p['passed']]
    first=min(wins) if wins else None
    losses=[p['docs'] for p in primary if first and p['docs']<first and not p['passed']]
    declared=[{'docs':p['docs'],'passed':p['decision']['versus_all_declared_cpu_configs']['performance_target_met_at_this_scale'],
        'cpu_roles':p['cpu_roles'],'session':'fresh_small_medium'} for p in measured]
    declared.append({'docs':reused['docs'],'passed':reused['decision']['versus_frozen_cpus']['performance_target_met_at_this_scale'],
        'cpu_roles':['cpu','cpu_directory'],'session':'reused_previous_10m'})
    declared.sort(key=lambda p:p['docs']);declared_wins=[p['docs'] for p in declared if p['passed']]
    return {'fresh_coverage_complete':len(measured)==len(points),'fresh_winning_docs_against_all_declared_configs':fresh_wins,
        'first_observed_winning_docs_under_declared_cpu_policy':min(declared_wins) if declared_wins else None,
        'declared_cpu_policy_points_with_session_origin':declared,
        'first_observed_winning_docs_against_common_frozen32_pair':first,
        'previous_observed_nonwinning_docs_against_common_pair':max(losses) if losses else None,
        'common_pair_points_with_session_origin':primary,
        'nonmonotonic_common_pair_results':any(a['passed'] and not b['passed'] for a,b in zip(primary,primary[1:])),
        'scope':'Only observed grid points.10m is from a previous session, not a fresh call. Single-thread reference exists only at<=65536. No interpolation, universal small-scale win or business-data acceptance.'}


def accepted_evidence(a):
    path=a.accepted/'report.json';old=json.loads(path.read_text(encoding='utf-8'))
    r.require(old.get('status')==bitmap.PASS and old['docs']==10000000,'Completed retained10m bitmap run required')
    r.require(old['source_sha256']==g.checks.sources(bitmap.OWN+x.OWN+x.HOST),'Tested bitmap/large source bytes changed')
    expected={name:bitmap.simd.hashlib.sha256(raw).hexdigest() for name,raw in bitmap.source.candidate_sources(a.original).items()}
    r.require(old['candidate']['metadata']['source_sha256']==expected,'Installed bitmap candidate differs from tested source')
    proofs={str(path):digest(path)}
    for key in ('previous_sha256','candidate_sha256'):
        bitmap.simd.close.scale.unchanged(old[key],'Tested artifacts changed');proofs.update(old[key])
    r.require(bitmap.decision(old['per_round'])==old['decision'] and old['decision']['retain_candidate'],'Accepted bitmap gate mismatch')
    prior={}
    for row in old['per_round']:
        for role in ('cpu','cpu_directory','baseline','candidate'):
            link=row[role];p=Path(link['directory'])/'report.json'
            r.require(digest(p)==link['report_sha256'],'Accepted process report changed');proofs[str(p)]=digest(p)
            result=json.loads(p.read_text(encoding='utf-8'));binary=Path(result['command'][0])
            r.require(result['summary']==link['summary'] and digest(binary)==result['runner_sha256'],'Accepted binary/summary changed')
            proofs[str(binary.resolve())]=result['runner_sha256']
            prior.setdefault(role,result)
    large_path=Path(old['baseline_large_report']);large=json.loads(large_path.read_text(encoding='utf-8'))
    r.require(large['status']=='actual_large_search_target_met_on_declared_data' and str(large_path) in proofs and digest(large_path)==proofs[str(large_path)],'Original scale evidence changed')
    work=old['workload'];r.require(work['dataset_kind']=='uniform_synthetic_not_MT_approved','Same declared synthetic family required')
    r.require(work['queries']==8 and work['query_length']==3 and work['k']==10,'FixedQ8/three words/K10 required')
    cases={c:work['cases'][c]['terms'] for c in ('A','B')}
    r.require(cases==large['fixed_query_cases'],'Fixed A/B IDs mismatch')
    return old,large,prior,proofs


def command_for(template,inputs,output,vendors=None,single=False):
    command=list(template['command']);npu=vendors is not None
    if npu:
        r.require(len(command)==14 and command[7:12]==['10','8','24','5','30'],'NPU template contract mismatch')
        command[6]=str(vendors);command[12:14]=[str(inputs/'A/expected_scores.bin'),str(inputs/'B/expected_scores.bin')]
    else:
        r.require(len(command)==12 and command[5:]==['32','10','32','8','24','5','30'],'Frozen CPU template mismatch')
        if single:command[5]=command[7]='1'
    command[1:5]=[str(inputs/'index'),str(inputs/'A.txt'),str(inputs/'B.txt'),str(output)]
    return command


def verify_single_cpu(path,inputs,description,index,role):
    # Validate the actual1/1 session directly; never relabel it32/32 to reuse
    # a validator, and never alter its original JSON or timing samples.
    r.require(role in ('cpu_single','cpu_directory_single'),'Explicit single-thread CPU role required')
    data=json.loads((path/'benchmark_session.json').read_text(encoding='utf-8'))
    wanted={'status':'search_benchmark_calls_completed_checks_pending','backend':'cpu','docs':description['docs'],
        'queries':8,'k':10,'partitions':1,'requested_threads':1,'actual_threads':1,'capacity_queries':8,'capacity_words':24,
        'input_contract':'raw_grouped_INT32_word_IDs_in_CPU_memory','timer_policy':g.checks.TIMER,'index_load_batches':1,
        'index_bytes':index['index_bytes'],'index_readback_byte_identical':True,'warmup_cycles':5,'warmup_search_calls':10,
        'measured_cycles':30,'measured_search_calls':60,'measured_topk_bit_checks':60,'checkpoint_search_calls':4,
        'measured_file_io_calls':0,'measured_debug_download_calls':0,'measured_poison_calls':0,
        'diagnostic_checkpoints_outside_measurement':True,'fp32_flags':'-O3 -march=native -ffp-contract=off -fno-fast-math'}
    r.require(all(type(data.get(k)) is type(v) and data[k]==v for k,v in wanted.items()),'Actual single-thread CPU contract mismatch')
    sources=g.checks.CPU_SOURCES if role=='cpu_single' else g.HOST_SOURCES
    r.require(data['host_source_sha256']==g.checks.sources(sources),'Compiled single-thread binary source mismatch')
    samples=data['samples'];r.require(len(samples)==60,'Incomplete single-thread samples')
    for pos,item in enumerate(samples):
        cycle,slot=divmod(pos,2);case='B' if slot^(cycle&1) else 'A'
        r.require(item['cycle']==cycle and item['case']==case and math.isfinite(item['call_wall_ms']) and
            math.isfinite(item['internal_search_ms']) and 0<item['internal_search_ms']<=item['call_wall_ms'] and
            item['query_h2d_bytes']==item['topk_d2h_bytes']==0,'Actual single-thread sample order/boundary differs')
    hashes={}
    for checkpoint in g.checks.CHECKPOINTS:
        actual=path/checkpoint;gold=inputs/checkpoint[-1]
        for target,expected in (('scores.bin','expected_scores.bin'),('top_ids.bin','expected_top_ids.bin'),('top_scores.bin','expected_top_scores.bin')):
            r.require(x.same_stream(actual/target,gold/expected),'Single-thread CPU ordered FP32/ranking differs')
        for p in actual.iterdir():
            if p.is_file():hashes[str(p.relative_to(path))]=digest(p)
    for case in ('A','B'):
        r.require(x.same_stream(path/('before_'+case)/'scores.bin',path/('after_'+case)/'scores.bin'),'Single-thread score bits changed')
    return {'session':data,'benchmark_session_sha256':digest(path/'benchmark_session.json'),'output_sha256':hashes,
        'summary':{c:g.checks.summarize([s['call_wall_ms'] for s in samples if s['case']==c],8) for c in ('A','B')}}


def pair_check(left,right):
    for checkpoint in g.checks.CHECKPOINTS:
        a=left/checkpoint;b=right/checkpoint
        for p in a.iterdir():
            if p.suffix=='.bin':r.require(x.same_stream(p,b/p.name),'Paired full scores/tree bits differ at this scale')


def run(a):
    r.require(sys.platform=='linux' and not a.output.exists() and a.output.is_relative_to(ROOT/'results'),'A5 Linux and fresh results output required')
    r.require(not a.output.is_relative_to(a.accepted) and not a.accepted.is_relative_to(a.output),'Independent results output required')
    old,large,prior,proofs=accepted_evidence(a);work=old['workload'];device=old['device'];cases={c:work['cases'][c]['terms'] for c in ('A','B')}
    seed=work['generation']['seed'];vocab=work['generation']['vocab'];own=g.checks.sources(OWN)
    # No10m work is scheduled. Account for all retained new checkpoints and new small inputs.
    reusable_docs={t['docs'] for t in large['tiers'] if t.get('status')=='six_rounds_checked'}
    disk_need=sum(n*4*8*4*6*(2+len(cpu_roles(n))) for n in a.sizes)+sum(n*(48*8+4+8*8) for n in a.sizes if n not in reusable_docs)+1024**3
    r.require(disk_need<shutil.disk_usage(a.output.parent).free,'Disk budget insufficient for all immutable sweep checkpoints')
    a.output.mkdir(parents=True)
    report={'status':'running','source_sha256':own,'previous_sha256':proofs,'device':device,'planned_fresh_sizes':a.sizes,
        'scope':'No rebuild/install or old correctness rerun. FixedQ8/three words/K10, full Search, six paired rounds per fresh size.',
        'cpu_policy':'Both original32/32 binaries remain; add SAME binaries at1/1 only for<=65536. Report common-pair and stronger all-config gates separately.',
        'disk_budget_bytes':disk_need,'tiers':[],'reused_10m':{'docs':old['docs'],'status':'reused_previous_session_not_rerun',
            'report':str(a.accepted/'report.json'),'report_sha256':digest(a.accepted/'report.json'),
            'decision':old['decision'],'workload':work,'per_round':old['per_round']}}
    def save():r.write_json(a.output/'report.json',report)
    def check():
        bitmap.simd.close.scale.unchanged(proofs,'Accepted source/artifacts/evidence changed')
        r.require(g.checks.sources(OWN)==own,'Sweep or accepted implementation source changed')
    save()
    try:
        report['host_context']={'allowed_cpu_affinity':sorted(os.sched_getaffinity(0)),
            'openmp_environment':{k:v for k,v in os.environ.items() if k.startswith(('OMP_','GOMP_'))}}
        report['management_before']=x.accepted.prior.management_snapshot(a.output,'sweep_before');save()
        runner=Path(prior['candidate']['command'][0]);vendors={role:Path(prior[role]['command'][6]) for role in ('baseline','candidate')}
        g.execute([runner,'--memory',device,a.output/'device_memory.json'],a.output/'memory.log',g.environment(vendors['candidate']))
        report['device_memory_probe']=json.loads((a.output/'device_memory.json').read_text(encoding='utf-8'));save();check()
        for n in a.sizes:
            print(f'Scale coverage N={n}; Q8/three words/K10 fixed.10m evidence reused, not rerun.',flush=True)
            matches=[t for t in large['tiers'] if t['docs']==n and t.get('status')=='six_rounds_checked']
            if matches:
                original=matches[0];link=original['per_round'][0]['npu'];path=Path(link['directory'])/'report.json'
                r.require(digest(path)==link['report_sha256'],'Original scale report changed');proofs[str(path)]=digest(path)
                previous=json.loads(path.read_text(encoding='utf-8'));inputs=Path(previous['command'][1]).parent
                desc,index=corpus.validate(inputs);r.require(desc==original['workload'],'Reused original scale input differs')
                input_origin='reused_original_large_scale_index_and_oracle'
            else:
                inputs=a.output/'inputs'/str(n);corpus.prepare(inputs,n,vocab,seed,cases)
                desc,index=corpus.validate(inputs);input_origin='new_prefix_of_same_synthetic_family'
            r.require(desc['docs']==n and desc['generation']['seed']==seed and desc['generation']['vocab']==vocab and
                {c:desc['cases'][c]['terms'] for c in ('A','B')}==cases,'Scale changed data rule or queries')
            proofs[str(inputs/'preparation_report.json')]=digest(inputs/'preparation_report.json')
            plan=x.layout(n,vocab,index['posting_count']);tier={'docs':n,'status':'prepared','input_origin':input_origin,
                'input_directory':str(inputs),'workload':desc,'memory_plan':plan,'cpu_roles':list(cpu_roles(n)),'per_round':[]}
            report['tiers'].append(tier);save()
            available=x.host_available_bytes()
            if plan['cpu_peak_known_bytes']+512*1024**2>=available*4//5:
                tier.update(status='not_measured_host_budget',available_host_bytes=available);save();continue
            memory_path=a.output/f'memory_N{n}.json'
            g.execute([runner,'--memory',device,memory_path],a.output/f'memory_N{n}.log',g.environment(vendors['candidate']))
            memory=json.loads(memory_path.read_text(encoding='utf-8'));tier['device_memory_probe']=memory
            if plan['known_device_bytes_upper_bound']+plan['debug_score_bytes']+1024**3>memory['free']*4//5:
                tier.update(status='not_measured_device_budget');save();continue
            def measure(role,path):
                path.mkdir(parents=True);actual=path/'actual';npu=role in vendors;single=role.endswith('_single')
                base_role=role.removesuffix('_single') if single else role
                command=command_for(prior[base_role],inputs,actual,vendors[role] if npu else None,single)
                check();g.execute(command,path/'run.log',g.environment(vendors[role]) if npu else os.environ.copy(),timeout=3600)
                result=verify_single_cpu(actual,inputs,desc,index,role) if single else x.verify_run(actual,inputs,desc,index,'npu' if npu else role,device)
                result.update(command=command,runner_sha256=digest(Path(command[0])),log_sha256=digest(path/'run.log'));check()
                r.write_json(path/'report.json',result)
                print(f'N={n} {role} complete Search: '+json.dumps(result['summary'],ensure_ascii=False),flush=True)
                return {'directory':str(path),'summary':result['summary'],'report_sha256':digest(path/'report.json')}
            for i in range(6):
                corpus.validate(inputs);order=process_order(n,i);row={'round':i+1,'process_order':list(order)}
                for role in order:row[role]=measure(role,a.output/f'N{n}'/f'round{i+1}'/role)
                pair_check(Path(row['baseline']['directory'])/'actual',Path(row['candidate']['directory'])/'actual')
                tier['per_round'].append(row);save()
            corpus.validate(inputs);tier.update(status='six_rounds_checked',decision=point_decision(tier['per_round'],n));save()
            print(f'N={n} scale boundary decision: '+json.dumps(tier['decision'],ensure_ascii=False),flush=True)
        report['curve_summary']=curve_summary(report['tiers'],report['reused_10m'])
        report['management_after']=x.accepted.prior.management_snapshot(a.output,'sweep_after')
        complete=report['curve_summary']['fresh_coverage_complete'];report['status']=CHECKED if complete else INCOMPLETE
        check();save();print('Bitmap scale coverage: '+json.dumps(report['curve_summary'],ensure_ascii=False),flush=True)
        print('Exit0 means requested scale coverage/correctness complete; individual CPU/NPU wins are reported separately.',flush=True)
        return 0 if complete else 2
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',first_error=str(error));save()
        logs=list(a.output.rglob('*.log'));last=max(logs,key=lambda p:p.stat().st_mtime) if logs else None
        if last:print('Latest log '+str(last)+'\n'+last.read_text(encoding='utf-8',errors='replace')[-24000:],flush=True)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accepted',type=Path,default=ROOT/'results/score_bitmap_large_close1')
    p.add_argument('--original',type=Path,default=ROOT/'results/Bm25FusedScoreTopKUbFix1')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--sizes',type=int,nargs='+',default=list(SIZES));a=p.parse_args()
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    try:validate_sizes(a.sizes);return run(a)
    except (Exception,KeyboardInterrupt) as error:print('FAILED: '+str(error)+'; no fabricated device result.',file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
