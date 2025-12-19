#!/usr/bin/env python3
"""
简单的SPARQL端点连接测试脚本
用于验证SPARQL端点是否可访问，以及SPARQLToolAdapter是否能正常工作
"""

import sys
from SPARQLWrapper import SPARQLWrapper, JSON

def test_sparql_endpoint(endpoint_url: str = "http://210.75.240.141:18890/sparql"):
    """测试SPARQL端点连接"""
    print(f"测试SPARQL端点: {endpoint_url}")
    
    try:
        sparql = SPARQLWrapper(endpoint_url)
        sparql.setReturnFormat(JSON)
        sparql.setTimeout(10)
        
        # 执行一个简单的查询
        test_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1"
        sparql.setQuery(test_query)
        
        print("执行测试查询...")
        results = sparql.query().convert()
        
        bindings = results.get("results", {}).get("bindings", [])
        if bindings:
            print(f"✓ SPARQL端点连接成功！返回了 {len(bindings)} 条结果")
            return True
        else:
            print("⚠ SPARQL端点连接成功，但查询返回空结果")
            return True
            
    except Exception as e:
        print(f"✗ SPARQL端点连接失败: {e}")
        return False

def test_sparql_adapter():
    """测试SPARQLToolAdapter"""
    print("\n测试SPARQLToolAdapter...")
    
    try:
        from kg_r1.llm_agent.sparql_tool_adapter import SPARQLToolAdapter
        
        adapter = SPARQLToolAdapter(
            sparql_endpoint="http://210.75.240.141:18890/sparql",
            kg_top_k=10
        )
        print("✓ SPARQLToolAdapter初始化成功")
        
        # 测试实体注册
        test_topic_entities = {"m.02mjmr": "Barack Obama"}
        adapter.seed_registry_from_topic_entities(test_topic_entities)
        print("✓ 实体注册表初始化成功")
        
        return True
        
    except Exception as e:
        print(f"✗ SPARQLToolAdapter测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SPARQL后端连接测试")
    print("=" * 60)
    
    endpoint = sys.argv[1] if len(sys.argv) > 1 else "http://210.75.240.141:18890/sparql"
    
    # 测试1: SPARQL端点连接
    endpoint_ok = test_sparql_endpoint(endpoint)
    
    # 测试2: SPARQLToolAdapter
    adapter_ok = test_sparql_adapter()
    
    print("\n" + "=" * 60)
    if endpoint_ok and adapter_ok:
        print("✓ 所有测试通过！可以开始训练。")
        sys.exit(0)
    else:
        print("✗ 部分测试失败，请检查配置。")
        sys.exit(1)

