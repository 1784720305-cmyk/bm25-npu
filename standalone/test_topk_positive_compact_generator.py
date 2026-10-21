"""Fake actual-template/argument/source-isolation checks; no msopgen/CANN/NPU execution."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import create_topk_positive_compact as tool


def main():
    count=0
    def fails(call):
        nonlocal count
        try: call()
        except (ValueError,OSError): count+=1
        else: raise AssertionError('invalid template/argument accepted')
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='positive_compact_FAKE_TEMPLATE_',dir=ROOT/'results') as tmp:
        base=Path(tmp)
        for typed in (False,True):
            project=base/str(typed)
            def fake(command,**kwargs):
                assert command[0]=='FAKE_MSOPGEN_NEVER_RUN'
                assert command[command.index('-i')+1]==str(ROOT/'standalone/topk_operator.json')
                assert command[-1]==str(project) and command[command.index('-c')+1]=='ai_core-ascend950'
                project.mkdir(); (project/'build.sh').write_bytes(b'FAKE TEMPLATE ONLY')
                for relative in ('op_host/bm25_top_k.cpp','op_host/bm25_top_k_tiling.h','op_kernel/bm25_top_k.cpp'):
                    file=project/relative; file.parent.mkdir(exist_ok=True); file.write_bytes(b'FAKE ORIGINAL TEMPLATE')
                tool.r.write_json(project/'CMakePresets.json',{'configurePresets':[{'cacheVariables':{
                    'ASCEND_COMPUTE_UNIT':{'type':'STRING','value':'old'} if typed else 'old'}}]})
            with patch.object(tool,'generator_command',return_value=['FAKE_MSOPGEN_NEVER_RUN']),patch.object(tool.subprocess,'run',fake),redirect_stdout(io.StringIO()):
                tool.generate(project,'ai_core-ascend950')
            info=json.loads((project/'bm25_project.json').read_text())
            assert info['operator']=='topk_positive_compact' and info['vendor_name']=='bm25_topk_positive_compact'
            assert info['positive_capacity']==1024 and info['compaction_min_docs']==1025 and info['membership_mask_max_docs']==8192
            assert info['selection_rule']=='FP32_score_desc_original_ID_asc_signed_zero_equal'
            assert info['status']=='project_generated_not_compiled_not_device_verified'
            assert info['generator_command'][0]=='FAKE_MSOPGEN_NEVER_RUN'; count+=5
            for name in tool.SOURCES:
                assert info['source_sha256'][name]==hashlib.sha256((project/name).read_bytes()).hexdigest()
                assert (project/name).read_bytes()==(ROOT/'npu_prototype/topk_positive_compact'/name).read_bytes()
                assert (project/'bm25_template_backup'/name).read_bytes()==b'FAKE ORIGINAL TEMPLATE'; count+=3
            preset=json.loads((project/'CMakePresets.json').read_text())['configurePresets'][0]['cacheVariables']
            assert preset['ASCEND_COMPUTE_UNIT']['value']=='ascend950' and preset['vendor_name']['value']=='bm25_topk_positive_compact'; count+=1
        with patch.object(tool.subprocess,'run') as child:
            fails(lambda:tool.generate(base/'False','ai_core-ascend950'))
            fails(lambda:tool.generate(base/'wrong_chip','ai_core-ascend910'))
            assert not child.called; count+=1
        malformed=base/'bad'; malformed.mkdir()
        fails(lambda:tool.populate(malformed))
        (malformed/'build.sh').write_bytes(b'FAKE TEMPLATE')
        fails(lambda:tool.populate(malformed))
        for name in ('op_host/bm25_top_k.cpp','op_kernel/bm25_top_k.cpp'):
            path=malformed/name; path.parent.mkdir(exist_ok=True); path.write_bytes(b'KEEP ORIGINAL')
        tool.r.write_json(malformed/'CMakePresets.json',{'configurePresets':[]})
        fails(lambda:tool.populate(malformed))
        assert (malformed/'op_kernel/bm25_top_k.cpp').read_bytes()==b'KEEP ORIGINAL'; count+=1
        # Geometry-only coverage checks; not a simulated hardware scheduler.
        for q,cores in ((1,56),(57,56),(513,56),(513,1)):
            blocks=min(q,cores)
            tasks=[x for b in range(blocks) for x in range(b,q,blocks)]
            assert sorted(tasks)==list(range(q)) and len(set(tasks))==q; count+=1
        kernel=(ROOT/'npu_prototype/topk_positive_compact/op_kernel/bm25_top_k.cpp').read_text()
        host=(ROOT/'npu_prototype/topk_positive_compact/op_host/bm25_top_k.cpp').read_text()
        assert 'docCount > kCapacity && docCount <= kMaskDocs' in kernel
        assert 'compact = positiveTotal <= kCapacity;' in kernel
        assert 'kScratchBytes == 16704' in kernel and 'kScratchBytes = 16704' in host
        assert 'topScores[query*k + lane] = valid ? scores[base + id]' in kernel
        assert 'equalHead = ReadId(scratch, doc, compact);' in kernel
        assert 'doc = equalPosition + sourceStep' in kernel; count+=6
    print(f'PASS: {count} fake-template/source-backup/metadata/rejection/grid-coverage/source-invariant checks. '
          'No generator execution, C++ compilation, SIMT synchronization or device/performance proof.')


if __name__=='__main__': main()
