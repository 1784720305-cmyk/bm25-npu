"""Audit actual C++ constexpr UB layout versus launch declaration; not compiled/device proof."""
import ast
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def number(expression,values):
    expression=re.sub(r'sizeof\s*\(\s*uint32_t\s*\)','4',expression)
    expression=re.sub(r'\b(0x[0-9a-fA-F]+|[0-9]+)[uUlL]+\b',r'\1',expression)
    def visit(node):
        if isinstance(node,ast.Constant) and type(node.value) is int: return node.value
        if isinstance(node,ast.Name): return values[node.id]
        if isinstance(node,ast.BinOp):
            a,b=visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Add): return a+b
            if isinstance(node.op,ast.Sub): return a-b
            if isinstance(node.op,ast.Mult): return a*b
            if isinstance(node.op,(ast.Div,ast.FloorDiv)): return a//b
        raise ValueError('Unsupported constexpr syntax')
    return visit(ast.parse(expression.strip(),mode='eval').body)


def constants(source):
    source=re.sub(r'//[^\n]*|/\*.*?\*/','',source,flags=re.S)
    values={}
    for declaration in re.findall(r'constexpr\s+uint32_t\s+([^;]+);',source):
        for item in declaration.split(','):
            name,expression=item.split('=',1); values[name.strip()]=number(expression,values)
    return values


def audit(regions,declared):
    if type(declared) is not int or declared <= 0 or declared%32: raise ValueError('Invalid dynamic UB declaration')
    end=0
    for name,start,stop in regions:
        if not (start == end and start < stop <= declared): raise ValueError('UB out of declared bounds: '+name)
        end=stop
    return end


def main():
    host=(ROOT/'npu_prototype/fused_score_topk/op_host/bm25_fused_score_top_k.cpp').read_text(encoding='utf-8')
    kernel=(ROOT/'npu_prototype/fused_score_topk/op_kernel/bm25_fused_score_top_k.cpp').read_text(encoding='utf-8')
    h,k=constants(host),constants(kernel)
    launch=re.findall(r'SetLocalMemorySize\(\s*([^)]*)\)',host)
    assert len(launch)==1; declared=number(launch[0],h)
    regions=[('scores',0,4*k['kMaxDocs']),('begins',4*k['kBegins'],4*k['kEnds']),
        ('ends',4*k['kEnds'],4*k['kTermIds']),('term_ids',4*k['kTermIds'],4*k['kScoreControl']),
        ('score_control',4*k['kScoreControl'],4*k['kSelection']),
        ('selection',4*k['kSelection'],4*k['kSelection']+k['kSelectScratchBytes'])]
    assert audit(regions,declared)==h['kBufferBytes']==k['kBufferBytes']==50816
    assert h['kDynamicUbBytes']==declared==65536
    assert k['kScoreControl']*4==34048 and (k['kScoreControl']+8)*4==34080
    assert 'static_assert(kDynamicUbBytes >= kBufferBytes' in host
    assert 'ubBytes < kDynamicUbBytes+kMinDCacheBytes+kCompilerReservedBytes' in host
    assert h['kMinDCacheBytes']==32768 and h['kCompilerReservedBytes']==8192
    checks=7
    for insufficient in (32768,34048,34080,49152,50815,0,-1,True):
        try: audit(regions,insufficient)
        except ValueError: checks+=1
        else: raise AssertionError('Accepted insufficient/malformed declared UB')
    for enough in (50816,51200,65536): assert audit(regions,enough)==50816; checks+=1
    # Corrupt actual region boundaries: detect holes, overlap and an overshooting selection end.
    for slot,newstart,newend in ((1,32764,33280),(4,34052,34112),(5,34112,65540)):
        bad=regions[:]; name,unused_start,unused_end=bad[slot]; bad[slot]=(name,newstart,newend)
        try: audit(bad,declared)
        except ValueError: checks+=1
        else: raise AssertionError('Accepted broken UB subregion')
    required=declared+h['kMinDCacheBytes']+h['kCompilerReservedBytes']
    assert required==106496 and required-1 < required and 256*1024 >= required; checks+=1
    # Old declaration fails even though the device reports a much larger physical UB.
    assert 256*1024 >= 50816+32768 and 32768 < regions[4][1]; checks+=1
    print(f'PASS: {checks} source-derived UB-layout/launch-size/cache-separation/regression checks. '
          'Historical32768-byte declaration rejected,50816-byte buffer covered by65536-byte declaration. '
          'No CANN compilation or device execution.')


if __name__=='__main__': main()
