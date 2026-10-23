"""Affected resource-copy tests only. Compiler subprocesses are MOCKED; no CANN/device claim."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import fused_resource_report as res

# Actual generated A5/CANN 9.2 template shared in the terminal summary.
ACTUAL='''if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    npu_op_kernel_options(ascendc_kernels ALL OPTIONS -g -O0)
endif()
npu_op_kernel_sources(ascendc_kernels
    KERNEL_DIR ./
)
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/
    TILING_LIBRARY cust_optiling
)
npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)
'''
CONFIG='CMAKE_BUILD_TYPE:STRING=Release\nASCEND_COMPUTE_UNIT:STRING=ascend950\nCMAKE_CXX_FLAGS:STRING=\n'


def snapshot(directory):
    return {p.relative_to(directory).as_posix():p.read_bytes() for p in directory.rglob('*') if p.is_file()}


def project(directory,template=ACTUAL):
    for name in res.fused.generator.SOURCES+('CMakePresets.json','build.sh'):
        path=directory/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'MOCK ORIGINAL\n')
    (directory/'op_kernel/CMakeLists.txt').write_bytes(template.encode('utf-8'))
    (directory/'build_out').mkdir()
    (directory/'build_out/CMakeCache.txt').write_bytes(CONFIG.encode('utf-8'))
    return directory


def main():
    count=0
    def reject(action):
        nonlocal count
        try:action()
        except (ValueError,OSError,subprocess.SubprocessError):count+=1
        else:raise AssertionError('invalid resource-copy evidence accepted')
    with tempfile.TemporaryDirectory(prefix='fused_resource_MOCK_ONLY_',dir=ROOT/'results') as tmp, \
         redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
        base=Path(tmp);fake_root=base/'root';(fake_root/'results').mkdir(parents=True)
        original=project(fake_root/'results/original');before=snapshot(original)
        runner=base/'FAKE_RUNNER_NOT_EXECUTABLE';runner.write_bytes(b'MOCK')
        hardware=fake_root/'results/fused_search_256k_q8_close1_hardware.json'
        res.r.write_json(hardware,{'compiler_help':{'command':['/FAKE_COMPILER'],
            'resource_usage_option_lines':['--cce-res-usage Print resource usage']}})
        installation={'fused':{'device_object_sha256':{'MOCK.o':'not-real'}}}
        with patch.object(res,'ROOT',fake_root):
            output=fake_root/'results/copy';record=res.prepare(original,output)
            expected='\n# Resource diagnostic COPY ONLY; no precision/optimization/thread changes.\n'+ \
                'npu_op_kernel_options(ascendc_kernels ALL OPTIONS --cce-res-usage)\n'+ACTUAL
            assert (output/'op_kernel/CMakeLists.txt').read_bytes()==expected.encode('utf-8')
            assert record['compile_options_api']=='npu_op_kernel_options' and record['compile_options_target']=='ascendc_kernels'
            assert snapshot(original)==before and not (output/'build_out').exists();count+=3
            reject(lambda:res.prepare(original,output))
            reject(lambda:res.prepare(original,original/'nested'))
            reject(lambda:res.prepare(original,fake_root/'outside_results'))
            for i,template in enumerate((ACTUAL.replace('ascendc_kernels','unknown_target'),
                    ACTUAL.replace('npu_op_kernel_options','unknown_api'),ACTUAL+ACTUAL)):
                source=project(fake_root/f'results/unsupported{i}',template);target=fake_root/f'results/no_copy{i}'
                assert res.prepare(source,target)['status']=='unsupported_actual_template' and not target.exists();count+=1
            legacy=ACTUAL.replace('npu_op_kernel_options(ascendc_kernels','add_ops_compile_options(')
            source=project(fake_root/'results/legacy',legacy);record=res.prepare(source,fake_root/'results/legacy_copy')
            assert record['compile_options_api']=='add_ops_compile_options';count+=1
            source=project(fake_root/'results/already',ACTUAL.replace('-g -O0','--cce-res-usage'))
            reject(lambda:res.prepare(source,fake_root/'results/already_copy'))
            source=project(fake_root/'results/absolute')
            (source/'build.sh').write_bytes(('cd "'+str(source.resolve())+'"\n').encode())
            reject(lambda:res.prepare(source,fake_root/'results/absolute_copy'))
            # Symlink creation may require Windows privileges; always exercise the explicit rejection branch via MOCK.
            real_is_symlink=Path.is_symlink
            def symlink(path):return path==original/'build.sh' or real_is_symlink(path)
            with patch.object(Path,'is_symlink',symlink):
                reject(lambda:res.prepare(original,fake_root/'results/symlink_copy'))
            for mode in ('pass','saved_stage_only','no_report','compile_failed','flags_changed','source_changed','option_unconfirmed'):
                directory=fake_root/'results'/mode;commands=[]
                def build(command,**kwargs):
                    commands.append(command);copy=kwargs['cwd']
                    assert command==['bash','build.sh'] and copy==directory/'project_copy'
                    assert kwargs['timeout']==900
                    assert kwargs['env']['ASCENDC_BUILD_LOG_DIR']==str((directory/'compiler_logs').resolve())
                    assert os.environ['ASCENDC_BUILD_LOG_DIR']=='PARENT_VALUE_UNCHANGED'
                    assert '--cce-res-usage' in (copy/'op_kernel/CMakeLists.txt').read_text()
                    (copy/'build_out').mkdir()
                    (copy/'build_out/CMakeCache.txt').write_bytes(
                        (CONFIG.replace('Release','Debug') if mode=='flags_changed' else CONFIG).encode())
                    if mode=='source_changed':(copy/res.fused.generator.SOURCES[0]).write_bytes(b'UNEXPECTED CHANGE')
                    if mode!='saved_stage_only' and mode!='no_report':
                        kwargs['stdout'].write(b'FusedScoreTopKSimt\nRegisters count: 70\nStack size: 0\n')
                    if mode=='saved_stage_only':
                        saved=Path(kwargs['env']['ASCENDC_BUILD_LOG_DIR'])/'ascend950/compile.log_success'
                        saved.parent.mkdir()
                        saved.write_bytes(b'[BISHENG] Function properties for _Z18FusedScoreTopKSimt_simt_entry: Stack size: 0 bytes, Used register number: 70\n')
                    return SimpleNamespace(returncode=7 if mode=='compile_failed' else 0)
                with patch.object(res.fused,'installed',return_value=installation), \
                     patch.object(res.shutil,'which',return_value=None if mode=='option_unconfirmed' else '/FAKE_COMPILER'), \
                     patch.object(res.subprocess,'run',build),patch.dict(os.environ,{'ASCENDC_BUILD_LOG_DIR':'PARENT_VALUE_UNCHANGED'}):
                    if mode in ('compile_failed','flags_changed','source_changed'):
                        reject(lambda:res.run(original,directory,runner,base,base))
                        assert json.loads((directory/'report.json').read_text())['status']=='failed'
                    else:
                        record=res.run(original,directory,runner,base,base)
                        if mode in ('pass','saved_stage_only','no_report'):
                            assert record['status']=='resource_diagnostic_compile_completed' and record['accepted_artifacts_unchanged']
                            assert record['configuration']['CMAKE_BUILD_TYPE']=='Release'
                            assert not record['compiler_logging']['parent_environment_modified']
                            if mode=='pass':assert any('Registers count: 70' in x for x in record['resource_context_lines'])
                            elif mode=='saved_stage_only':
                                assert record['fused_vf_reports']==[{'path':'compiler_logs/ascend950/compile.log_success',
                                    'function':'_Z18FusedScoreTopKSimt_simt_entry','stack_size_bytes':0,'used_register_number':70}]
                                assert len(record['compiler_log_records'])==2
                            else:assert record['fused_vf_reports']==[] and record['resource_context_lines']==[]
                        else:assert record['status']=='resource_option_not_confirmed'
                        count+=1
                assert commands==([] if mode=='option_unconfirmed' else [['bash','build.sh']])
                assert snapshot(original)==before;count+=1
            # A foreign function, missing register field and non-UTF8 logs must not turn UNKNOWN into zero.
            logs=fake_root/'results/parser';(logs/'compiler_logs').mkdir(parents=True)
            (logs/'resource_build.log').write_bytes(b'[BISHENG] Function properties for Other: Stack size: 0 bytes, Used register number: 1\n')
            (logs/'compiler_logs/partial_success').write_bytes(
                b'[BISHENG] Function properties for _ZFusedScoreTopKSimt_simt_entry: Stack size: 8 bytes\n')
            (logs/'compiler_logs/not_text_success').write_bytes(b'\xff\xfe\0')
            found,reports=res.compiler_reports(logs)
            assert len(reports)==1 and reports[0]['stack_size_bytes']==8 and reports[0]['used_register_number'] is None
            assert any('unread_reason' in x for x in found);count+=2
            # Resource-only CLI routes straight to the copy build; no profiler/Search child is reachable here.
            for status,code in (('resource_diagnostic_compile_completed',0),('resource_template_unavailable',1)):
                with patch.object(res.sys,'platform','linux'),patch.object(res,'run',return_value={'status':status}) as mocked:
                    assert res.main(['--output',str(fake_root/'results/cli')])==code
                    assert mocked.call_count==1 and mocked.call_args.args[1]==fake_root/'results/cli';count+=1
    print(f'PASS: {count} affected resource-template/copy/Release/first-error/CLI checks; MOCK compiler only. No CANN build, profiler or NPU execution.')


if __name__=='__main__':main()
