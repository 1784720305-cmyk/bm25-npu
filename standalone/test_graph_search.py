"""Portable checks for generated directory sources, data/oracles and performance gates. No CANN claims."""
from array import array
from copy import deepcopy
from pathlib import Path
import bisect
import json
import random
import re
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import graph_search as g


class GraphSearchTests(unittest.TestCase):
    def test_only_declared_kernel_changes(self):
        original=ROOT/'npu_prototype/fused_score_topk'
        old=g.simd.source.candidate_sources(original)
        new=g.source.candidate_sources(original)
        self.assertEqual(set(old),set(new))
        for name in old:
            if name=='op_host/bm25_fused_score_top_k.cpp':
                self.assertEqual(new[name].decode().replace('length[0] != vocab*(*shards+1)','length[0] != vocab+1'),old[name].decode())
            elif name=='op_kernel/bm25_fused_score_top_k.cpp':
                text=new[name].decode()
                self.assertIn('termOffsets[entry+1]',text)
                self.assertNotIn('LowerBound(postingDocs,',text)
                text=text.replace('                const uint64_t entry = static_cast<uint64_t>(term)*(shards+1)+task%shards;\n'
                    '                const int64_t begin = termOffsets[entry], end = termOffsets[entry+1];',
                    '                const int64_t begin = termOffsets[term], end = termOffsets[term+1];')
                text=text.replace('                    begins[thread] = begin; ends[thread] = end;',
                    '                    const int64_t lo = LowerBound(postingDocs, begin, end, first);\n'
                    '                    begins[thread] = lo; ends[thread] = LowerBound(postingDocs, lo, end, last);')
                self.assertEqual(text,old[name].decode())
            else:self.assertEqual(old[name],new[name])

    def test_directory_membership_for_empty_and_uneven_partitions(self):
        rng=random.Random(260126)
        for n in (1,3,41,96,8193):
            for shards in (1,2,7,32):
                if shards>n:continue
                for count in (0,1,min(n,20),n):
                    postings=sorted(rng.sample(range(n),count))
                    boundaries=[bisect.bisect_left(postings,n*s//shards) for s in range(shards+1)]
                    rebuilt=[]
                    for s in range(shards):
                        piece=postings[boundaries[s]:boundaries[s+1]]
                        self.assertTrue(all(n*s//shards<=d<n*(s+1)//shards for d in piece))
                        rebuilt.extend(piece)
                    self.assertEqual(rebuilt,postings)

    def test_real_small_fixture_roundtrip_and_ordered_oracle(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            plan=g.edge_inputs(root)
            index,values=g.r.validate_index(root/'index')
            lines=plan.read_text().splitlines()
            self.assertEqual(len(lines),8)
            previous={}
            for line in lines:
                label,directory=line.split('\t');directory=Path(directory)
                record,queries=g.r.validate_queries(directory,index)
                expected=g.r.diagnostic_scores(index,values,queries)
                raw=(directory/'expected_scores.bin').read_bytes()
                self.assertEqual(raw,g.r.encode(expected))
                self.assertEqual(len(raw),2*96*4)
                key=tuple(map(tuple,queries))
                if key in previous:self.assertEqual(raw,previous[key])
                previous[key]=raw
                self.assertEqual(record['k'],10)
            self.assertTrue(any(len(row)==64 for key in previous for row in key))

    def rows(self,cpu=.070,cpu_directory=.064,npu=.050,p95=.055):
        def measurement(m,p):return {'summary':{name:{'median_batch_ms':m,'p95_batch_ms':p} for name in ('A','B')}}
        return [{'cpu':measurement(cpu,.080),'cpu_directory':measurement(cpu_directory,.072),
            'graph_directory':measurement(npu,p95)} for _ in range(6)]

    def test_gate_requires_faster_than_fastest_cpu(self):
        self.assertTrue(g.summary_gate(self.rows())['performance_target_met'])
        self.assertFalse(g.summary_gate(self.rows(npu=.065))['performance_target_met'])
        self.assertFalse(g.summary_gate(self.rows(cpu=.049))['performance_target_met'])

    def test_gate_requires_all_six_and_p95(self):
        rows=self.rows();rows[3]['graph_directory']['summary']['B']['median_batch_ms']=.080
        self.assertFalse(g.summary_gate(rows)['performance_target_met'])
        self.assertFalse(g.summary_gate(self.rows(p95=.073))['performance_target_met'])
        with self.assertRaises(ValueError):g.summary_gate(rows[:5])

    def test_target_is_not_substituted_for_relative_success(self):
        result=g.summary_gate(self.rows(cpu=.050,cpu_directory=.049,npu=.054))
        self.assertFalse(result['performance_target_met'])
        self.assertTrue(result['cases']['A']['engineering_55us_target_met'])

    def test_cpp_and_python_provenance_lists_match(self):
        cmake=(ROOT/'npu_prototype/graph_search_host/CMakeLists.txt').read_text()
        block=cmake[cmake.index('set(SOURCES\n')+len('set(SOURCES\n'):cmake.index('\nset(RECORD')]
        names=tuple(block.rstrip().rstrip(')').split())
        self.assertEqual(names,g.HOST_SOURCES)
        self.assertTrue(all((ROOT/name).is_file() for name in names))

    def test_request_timer_and_capture_structure(self):
        text=(ROOT/'npu_prototype/graph_search_host/planned_search.h').read_text()
        body=text[text.index('    PlannedResult Search('):text.index('    void PoisonForDiagnostic(')]
        self.assertLess(body.index('Clock::now()'),body.index('layout.Pack('))
        self.assertLess(body.index('aclrtSynchronizeStream('),body.index('Receive(logical)'))
        self.assertLess(body.index('Receive(logical)'),body.rindex('Clock::now()'))
        capture=text[text.index('            Require(aclmdlRICaptureBegin('):text.index('            resources.capturing=false; resources.modelValid=true;')]
        self.assertIn('Upload(); Execute(prepared,workspace); Download();',capture)
        self.assertNotIn('aclrtSynchronizeStream',capture)
        self.assertIn('normalPrepareCalls+=2',body)
        self.assertIn('resources.model,resources.runtime.stream',body)


if __name__=='__main__':unittest.main()
