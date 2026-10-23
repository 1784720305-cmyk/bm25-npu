"""Affected instrumentation layout, raw-cycle analysis and fail-stop checks. No CANN/device execution."""
from array import array
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import profile_fused_phases as x
from standalone.test_fused_ub_contract import constants
from standalone.test_fused_resource_report import ACTUAL


def encode(values):
    a=array('Q',values)
    if sys.byteorder!='little':a.byteswap()
    return a.tobytes()


def trace(score=300,selection=700):
    return encode([v for task in range(256) for v in
        (10000*task,10000*task+3,10000*task+3+score,10000*task+3+score+selection)])


class PhaseTests(unittest.TestCase):
    def test_algorithm_and_barriers_retained(self):
        original=ROOT/'npu_prototype/fused_score_topk'
        old=x.t.candidate_sources(original);new=x.diagnostic_sources(original)
        before=old['op_kernel/bm25_fused_score_top_k.cpp'].decode()
        after=new['op_kernel/bm25_fused_score_top_k.cpp'].decode()
        # Entire selector before the score-layout marker is byte-identical except the one debug header.
        marker='// Three disjoint regions:'
        self.assertEqual(before.split(marker)[0],after.split(marker)[0].replace('\n#include "utils/debug/asc_time.h"',''))
        self.assertEqual(before.count('asc_syncthreads()'),after.count('asc_syncthreads()'))
        self.assertEqual(before.count('atomicAdd('),after.count('atomicAdd('))
        for line in ('const float numerator = tf*2.2F;','const float contribution = (weight*numerator)/(tf+norm);',
                     'const float total = scores[local]+contribution;',
                     'SelectLocal(memory, partialScores, partialIds, memory+kSelection, task, width, k);'):
            self.assertEqual(after.count(line),1)
        self.assertEqual(new['op_host/bm25_fused_score_top_k_tiling.h'],old['op_host/bm25_fused_score_top_k_tiling.h'])

    def test_timestamp_storage_alignment_and_disjointness(self):
        source=x.diagnostic_sources(ROOT/'npu_prototype/fused_score_topk')
        kernel=source['op_kernel/bm25_fused_score_top_k.cpp'].decode()
        c=constants(kernel)
        self.assertEqual((c['kThreads'],c['kBufferBytes'],c['kSelection']),(256,50848,8528))
        start=c['kScoreControl']+2;end=start+6
        self.assertEqual((start*4)%8,0)
        self.assertEqual(end,c['kScoreControl']+8)
        self.assertLess(end,c['kSelection'])
        self.assertEqual(kernel.count('phase['),6) # three clock writes, three final reads
        self.assertEqual(kernel.count('__asc_simt_vf::clock()'),4)
        # No new parameters, local arrays, dynamic UB, scan widths or launch-shape changes.
        host=source['op_host/bm25_fused_score_top_k.cpp'].decode()
        self.assertIn('kDynamicUbBytes = 64*1024',host)
        self.assertIn('(*debug != 0 && *debug != 1 && *debug != 2)',host)
        self.assertIn('queries > kLimit/((*shards)*8)',host)
        self.assertIn('static_cast<int64_t>(dims.queries)*dims.shards*8',host)

    def test_boundary_order_and_no_full_score_write_in_clock_mode(self):
        text=x.diagnostic_sources(ROOT/'npu_prototype/fused_score_topk')['op_kernel/bm25_fused_score_top_k.cpp'].decode()
        start=text.index('phase[1] = __asc_simt_vf::clock()')
        score=text.index('const float contribution =')
        middle=text.index('phase[2] = __asc_simt_vf::clock()')
        select=text.index('SelectLocal(memory, partialScores')
        barrier=text.index('asc_syncthreads(); // grid-stride next task')
        finish=text.index('const uint64_t finished = __asc_simt_vf::clock()')
        store=text.index('trace[task*4] =')
        self.assertTrue(start<score<middle<select<barrier<finish<store)
        self.assertIn('if (debug == 1) {\n            const uint64_t base =',text)

    def test_changed_frozen_anchor_rejected(self):
        old=x.t.candidate_sources(ROOT/'npu_prototype/fused_score_topk')
        for needle in ('const bool bad = control[0] != 0;','if (debug != 0) {','const uint32_t last = first+width;'):
            bad=dict(old);key='op_kernel/bm25_fused_score_top_k.cpp'
            bad[key]=bad[key].replace(needle.encode(),b'changed anchor',1)
            with self.subTest(needle=needle),patch.object(x.t,'candidate_sources',return_value=bad),self.assertRaises(ValueError):
                x.diagnostic_sources(Path('unused'))

    def test_per_task_deltas_are_same_clock_stream_only(self):
        rows=x.trace_rows(trace())
        self.assertEqual(len(rows),256)
        self.assertEqual(rows[32],dict(task=32,query=1,shard=0,clock_read_pair_cycles=3,score_cycles=300,
            selection_cycles=700,score_and_selection_cycles=1000,selection_fraction=.7))
        # A different arbitrary origin per task must not change either measured interval.
        values=[v for i in range(256) for v in (2**50+i*19,2**50+i*19+3,2**50+i*19+303,2**50+i*19+1003)]
        other=x.trace_rows(encode(values))
        self.assertEqual([r['score_cycles'] for r in rows],[r['score_cycles'] for r in other])
        self.assertEqual([r['selection_cycles'] for r in rows],[r['selection_cycles'] for r in other])

    def test_missing_wrapped_and_poisoned_clock_records_stop(self):
        for raw in (trace()[:-1],trace()+b'\0'*32,b'\xff'*8192,
            encode([0,3,3,900]*256),encode([4,3,300,900]*256),encode([0,3,300,299]*256)):
            with self.subTest(length=len(raw)),self.assertRaises(ValueError):x.trace_rows(raw)

    def test_zero_clock_pair_gap_does_not_fake_overhead_subtraction(self):
        rows=x.trace_rows(encode([0,0,300,900]*256))
        self.assertEqual(rows[0]['clock_read_pair_cycles'],0)
        self.assertEqual(rows[0]['score_cycles'],300)
        self.assertEqual(rows[0]['selection_cycles'],600)

    def test_cycle_statistics_do_not_pick_lowest(self):
        s=x.summarize(list(range(1,21)))
        self.assertEqual(s,dict(median=10.5,p95=19,min=1,max=20,count=20))
        for values in ([],[float('nan')],[-1],[float('inf')]):
            with self.assertRaises(ValueError):x.summarize(values)

    def test_actual_clock_header_hash_and_namespace_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);header=root/'asc_time.h';impl=root/'asc_time_simt_impl.h'
            text='namespace __asc_simt_vf { __SIMT_DEVICE_FUNCTIONS_DECL__ inline uint64_t clock(void); }\n#include "impl/utils/debug/asc_time_simt_impl.h"\n'
            header.write_bytes(text.encode());impl.write_bytes(b'actual implementation placeholder, not executed')
            sha=hashlib.sha256(header.read_bytes()).hexdigest()
            with patch.object(x,'CLOCK_SHA',sha),redirect_stdout(io.StringIO()):
                result=x.clock_header(root)
            self.assertEqual(result['actual_header_sha256'],sha)
            self.assertEqual(result['implementation_headers'],{str(impl.resolve()):x.digest(impl)})
            with self.assertRaises(ValueError):x.clock_header(root)
            header.write_text(text.replace('__asc_simt_vf','__asc_aicore'))
            with patch.object(x,'CLOCK_SHA',x.digest(header)),self.assertRaises(ValueError):x.clock_header(root)
            header.write_bytes(text.encode());impl.unlink()
            with patch.object(x,'CLOCK_SHA',sha),self.assertRaises(ValueError):x.clock_header(root)

    def test_analyzer_all_rounds_same_call_shares_and_source_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            actual=Path(directory)
            for name in ('A','B'):
                (actual/name).mkdir()
                for filename in x.fused.chain.FILES:(actual/name/filename).write_bytes(b'placeholder')
                for cycle in range(10):
                    # Alternate call duration scales: computing ratios AFTER pooling/medians would be wrong.
                    score,selection=((300,700) if cycle%2==0 else (8000,2000))
                    (actual/f'cycle{cycle}_{name}.bin').write_bytes(trace(score,selection))
            (actual/'device_session.json').write_text('{}')
            index={'num_docs':262144,'index_bytes':94008}
            records={name:{'num_queries':8,'k':10} for name in ('A','B')}
            session=dict(status='phase_cycles_outputs_and_guards_checked',device=5,queries=8,docs=262144,k=10,shards=32,
                warmup=5,repeats=10,index_upload_batches=1,index_tensor_h2d_calls=5,index_upload_bytes=94008,
                index_readback_byte_identical=True,debug1_calls=2,clock_mode_calls=32,saved_trace_calls=20,
                every_clock_call_rank_bits_match_debug1=True,all_output_guards_passed=True,
                full_score_guard_check_policy='preflight_and_after_all_clock_calls',
                measured_calls_full_score_poison_or_download=False,
                host_source_sha256=x.close.checks.sources(x.HOST))
            with patch.object(x.close.checks,'check_output',return_value={'passed':True}) as checked:
                result=x.analyze(actual,session,index,records,{'A':{},'B':{}},5)
                self.assertEqual(checked.call_count,2)
                self.assertEqual(len(result['samples']),20)
                self.assertAlmostEqual(result['cases']['A']['per_call_selection_work_fraction']['median'],.45)
                self.assertEqual(result['cases']['B']['per_call_selection_work_fractions'],[.7,.2]*5)
                for key,value in (('saved_trace_calls',19),('device',2),('index_readback_byte_identical',False),('index_upload_batches',True)):
                    bad=copy.deepcopy(session);bad[key]=value
                    with self.subTest(key=key),self.assertRaises(ValueError):x.analyze(actual,bad,index,records,{'A':{},'B':{}},5)
                with patch.object(x.close.checks,'check_output',side_effect=ValueError('first ranking error')):
                    with self.assertRaisesRegex(ValueError,'first ranking error'):x.analyze(actual,session,index,records,{'A':{},'B':{}},5)
                (actual/'cycle9_B.bin').unlink()
                with self.assertRaises(ValueError):x.analyze(actual,session,index,records,{'A':{},'B':{}},5)

    def test_failed_compile_never_installs_or_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);a=SimpleNamespace(output=output,project=output/'old',vendors=output/'vendors')
            child=SimpleNamespace(returncode=1)
            with patch.object(x,'prepare',return_value={}),patch.object(x.subprocess,'run',return_value=child) as run:
                with self.assertRaisesRegex(ValueError,'编译失败'):x.build_install(a)
            self.assertEqual(run.call_count,1)
            self.assertEqual(run.call_args.args[0],['bash','build.sh'])

    def test_independent_project_actual_abi_and_object_identity(self):
        with tempfile.TemporaryDirectory(prefix='phase_MOCK_ONLY_',dir=ROOT/'results') as directory:
            base=Path(directory);original=base/'original';original.mkdir()
            for name in x.fused.generator.SOURCES:
                path=original/name;path.parent.mkdir(exist_ok=True,parents=True)
                path.write_bytes((ROOT/'npu_prototype/fused_score_topk'/name).read_bytes())
            (original/'op_kernel/CMakeLists.txt').write_bytes(ACTUAL.encode())
            x.r.write_json(original/'CMakePresets.json',{'configurePresets':[{'cacheVariables':{
                'ASCEND_COMPUTE_UNIT':{'type':'STRING','value':'ascend950'},
                'vendor_name':{'type':'STRING','value':'bm25_fused_score_topk'}}}]})
            x.r.write_json(original/'bm25_project.json',{'dynamic_ub_bytes':65536,'compute_unit':'ascend950'})
            (original/'build.sh').write_bytes(b'MOCK NOT A BUILD')
            tree=lambda path:{str(p.relative_to(path)):p.read_bytes() for p in path.rglob('*') if p.is_file()}
            before=tree(original);target=base/'copy';record=x.prepare(original,target)
            self.assertEqual(tree(original),before)
            self.assertEqual(record['vendor_name'],x.VENDOR)
            self.assertEqual((record['simt_threads'],record['explicit_ub_bytes'],record['dynamic_ub_bytes']),(256,50848,65536))
            with self.assertRaises(ValueError):x.prepare(original,target)
            with self.assertRaises(ValueError):x.prepare(original,original/'nested')
            vendor=base/x.VENDOR;(vendor/'op_api/include').mkdir(parents=True);(vendor/'op_api/lib').mkdir()
            header=vendor/'op_api/include/aclnn_bm25_fused_score_top_k.h'
            declarations='\n'.join('aclnnStatus '+n+'('+args+');' for n,args in x.fused.PROTOTYPES.items())
            header.write_bytes(declarations.encode());(vendor/'op_api/lib/libcust_opapi.so').write_bytes(b'MOCK NOT ELF')
            build=target/'build_out';build.mkdir();(build/header.name).write_bytes(header.read_bytes())
            (build/'libcust_opapi.so').write_bytes(b'MOCK NOT ELF')
            objects=vendor/'op_impl/ai_core/tbe/kernel/ascend950/bm25_fused_score_top_k';objects.mkdir(parents=True)
            (objects/'MOCK.o').write_bytes(b'MOCK NOT DEVICE');(build/'MOCK.o').write_bytes(b'MOCK NOT DEVICE')
            result=x.installation(vendor,target,original)
            self.assertEqual(result['vendor'],str(vendor.resolve()))
            header.write_bytes(declarations.replace('int64_t k','int32_t k').encode())
            with self.assertRaisesRegex(ValueError,'ABI改变'):x.installation(vendor,target,original)
            header.write_bytes(declarations.encode());(objects/'MOCK.o').write_bytes(b'changed object')
            with self.assertRaises(ValueError):x.installation(vendor,target,original)


if __name__=='__main__':
    unittest.main(verbosity=2)
