"""Synthetic repeated-ID CSV checks only. No real profiler/device evidence."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import profile_graph_directory as p


def fixture():
    n,q,k,s,v,m=262144,8,10,32,1000,90000
    directory=v*(s+1)*8
    session={'docs':n,'queries':q,'k':k,'partitions':s,'capacity_words':24,
        'directory_bytes':directory,'index_bytes':directory+8*m+4*n+4*v,'warmup_cycles':5,
        'samples':[{'cycle':i//2,'case':'B' if (i%2)^((i//2)&1) else 'A','call_wall_ms':.1} for i in range(20)]}
    summary=[];tasks=[]
    for pos,(phase,case,debug) in enumerate(p.rows.phases(session)):
        for stage,op in enumerate(p.OPERATORS):
            start=1000+pos*200+stage*45;duration=40 if stage==0 else 5
            # Normal graph nodes reuse their task IDs; each node is on a different stream.
            tid=str(10+stage) if not debug else str(1000+pos*2+stage)
            stream=str(3+stage) if not debug else '1'
            shapes=f'{v*(s+1)};{m};{m};{n};{v};{q+1};24' if stage==0 else f'{q},{s},{k};{q},{s},{k}'
            output=f'{q},{s},{k};{q},{s},{k};'+(f'{q},{n}' if debug else '1')
            summary.append({'Device_id':'5','Task ID':tid,'Stream ID':stream,'OP Type':op,
                'Task Type':'AI_VECTOR_CORE','Task Start Time(us)':str(start),'Task Duration(us)':str(duration),
                'Block Num':'56' if stage==0 else '8','Input Shapes':shapes,'Output Shapes':output})
            tasks.append({'Device_id':'5','stream_id':stream,'task_id':tid,'kernel_name':op+'_simulated',
                'kernel_type':'AI_VECTOR_CORE','task_time(us)':str(duration),'task_start(us)':str(start),
                'task_stop(us)':str(start+duration)})
    return summary,tasks,session


class ProfileGraphTests(unittest.TestCase):
    def test_repeated_ids_distinct_streams_and_exclusions(self):
        summary,tasks,session=fixture()
        result=p.analyze_tables(list(reversed(summary)),list(reversed(tasks)),session,5)
        self.assertEqual(result['own_operator_execution_rows'],78)
        self.assertEqual(result['measured_search_calls'],20)
        self.assertGreater(result['reused_task_id_occurrences'],0)
        self.assertEqual(result['cases']['A']['device_chain_span_us']['median_us'],50)
        self.assertEqual(result['cases']['A']['outside_compute_chain_us']['median_us'],50)

    def test_missing_occurrence_rejected(self):
        summary,tasks,session=fixture()
        with self.assertRaises(ValueError):p.analyze_tables(summary[:-1],tasks,session,5)
        with self.assertRaises(ValueError):p.analyze_tables(summary,tasks[:-1],session,5)

    def test_ambiguous_occurrence_rejected(self):
        summary,tasks,session=fixture();tasks.append(deepcopy(tasks[0]))
        with self.assertRaises(ValueError):p.analyze_tables(summary,tasks,session,5)

    def test_wrong_fixed_word_shape_rejected(self):
        summary,tasks,session=fixture();summary[0]['Input Shapes']=summary[0]['Input Shapes'].rsplit(';',1)[0]+';1'
        with self.assertRaises(ValueError):p.analyze_tables(summary,tasks,session,5)

    def test_wrong_directory_shape_rejected(self):
        summary,tasks,session=fixture();summary[0]['Input Shapes']='1001;'+summary[0]['Input Shapes'].split(';',1)[1]
        with self.assertRaises(ValueError):p.analyze_tables(summary,tasks,session,5)

    def test_host_device_wrong_pair_rejected(self):
        summary,tasks,session=fixture();session['samples'][0]['call_wall_ms']=.02
        with self.assertRaises(ValueError):p.analyze_tables(summary,tasks,session,5)

    def test_task_time_disagreement_rejected(self):
        summary,tasks,session=fixture();tasks[0]['task_time(us)']='41'
        with self.assertRaises(ValueError):p.analyze_tables(summary,tasks,session,5)

    def test_device_and_debug_shape_rejected(self):
        summary,tasks,session=fixture()
        with self.assertRaises(ValueError):p.analyze_tables(summary,tasks,session,4)
        summary[4]['Output Shapes']='8,32,10;8,32,10;1'
        with self.assertRaises(ValueError):p.analyze_tables(summary,tasks,session,5)


if __name__=='__main__':unittest.main()
