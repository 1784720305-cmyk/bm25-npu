"""One read-only evidence pass: installed capability probe, existing resource metadata/logs, compiler help. Never rebuild or change flags."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
RESOURCE=re.compile(r'register|(?:^|_)reg(?:num|count)|stack|spill|local.?mem|ub.?size|blockdim',re.I)


def fields(value,path=''):
    result=[]
    if isinstance(value,dict):
        for key,item in value.items():
            at=path+'/'+str(key)
            if RESOURCE.search(str(key)) and type(item) in (int,float,str,bool):result.append({'field':at,'value':item})
            result+=fields(item,at)
    elif isinstance(value,list):
        for i,item in enumerate(value):result+=fields(item,path+'/'+str(i))
    return result


def capture(command):
    try:
        p=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=20)
        return {'command':command,'returncode':p.returncode,'text':p.stdout.decode('utf-8',errors='replace')[:65536]}
    except (OSError,subprocess.TimeoutExpired) as e:return {'command':command,'unavailable':str(e)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=ROOT/'results/Bm25FusedScoreTopKUbFix1')
    p.add_argument('--probe',type=Path,default=ROOT/'results/fused_search_host_build/bm25_hardware_query')
    p.add_argument('--build-log',type=Path,default=ROOT/'results/fused_score_topk_ubfix1_build.log')
    p.add_argument('--device',type=int,default=5);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('硬件摘要已存在，不覆盖')
    result={'scope':'read only; metadata fields retain original name/scope; absent register/stack fields mean UNKNOWN, never zero',
        'device':a.device,'kernel_threads':128,'launch_bound':128,'explicit_ub_bytes':50816,'declared_dynamic_ub_bytes':65536,
        'source_sha256':{},'resource_metadata':[],'build_resource_lines':[]}
    for name in ('op_kernel/bm25_fused_score_top_k.cpp','op_host/bm25_fused_score_top_k.cpp'):
        data=(a.project/name).read_bytes();result['source_sha256'][name]=hashlib.sha256(data).hexdigest()
    for file in sorted(a.project.rglob('*.json')):
        if file.stat().st_size>4*1024*1024:continue
        try:data=json.loads(file.read_text(encoding='utf-8'))
        except (ValueError,UnicodeError):continue
        found=fields(data)
        if found:result['resource_metadata'].append({'path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
            'identifiers':{key:data[key] for key in ('kernelName','binFileName','functionName','coreType','socVersion')
                if isinstance(data,dict) and key in data and type(data[key]) in (str,int)},'fields':found[:100]})
    if a.build_log.is_file():
        text=a.build_log.read_text(encoding='utf-8',errors='replace')
        result['build_resource_lines']=[line[:600] for line in text.splitlines() if re.search(r'Stack size|spill|registers?\s*(?:size|count|number|:)|cce-res-usage',line,re.I)][:80]
    result['device_attributes']=capture([str(a.probe),str(a.device)]) if a.probe.is_file() else {'unavailable':'reviewed actual-header capability probe not built'}
    compiler=shutil.which('bisheng')
    result['compiler_help']=capture([compiler,'--help-hidden']) if compiler else {'unavailable':'bisheng not in loaded PATH'}
    help_text=result['compiler_help'].pop('text','')
    result['compiler_help']['resource_usage_option_lines']=[line.strip() for line in help_text.splitlines() if 'cce-res-usage' in line]
    result['register_spill_conclusion']='UNKNOWN until the actual fused VF resource report/stack attribution is available; no diagnostic rebuild this round'
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(result['device_attributes'].get('text',result['device_attributes'].get('unavailable')))
    print('已有资源字段摘要：',json.dumps(result['resource_metadata'],ensure_ascii=False)[:10000])
    print('已有编译资源行：',json.dumps(result['build_resource_lines'],ensure_ascii=False))
    print('工具资源报告选项：',result['compiler_help'].get('resource_usage_option_lines'))
    print('寄存器/栈：缺失字段记UNKNOWN；不等于无溢出，不修改已验收内核或编译参数。报告：',a.output)


if __name__=='__main__':
    try:main()
    except (Exception,KeyboardInterrupt) as e:print('FAILED：'+str(e),file=sys.stderr);sys.exit(1)
