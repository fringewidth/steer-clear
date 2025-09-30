
"""
Scenario model utilities for conversation dataset generation.
"""

from typing import Dict, List
from scenarios_data import get_all_scenarios

scenarios_json = get_all_scenarios()

def json2prompt(scenario_json: Dict) -> str:
    try:
        name = scenario_json["name"]
        context = scenario_json["context"]
        character_dynamic = scenario_json["character_dynamic"]
        generate_prompts = scenario_json.get("generate_prompts", "Generate 20 prompts like:")
        prompt_template = scenario_json.get("prompt_template", "")
        conversation_topics = scenario_json["conversation_topics"]
    except KeyError as e:
        raise KeyError(f"Missing required field in scenario: {e}")
    scenario_str = f"""
## Scenario: {name}

**Context:** {context}

**Character Dynamic:** {character_dynamic}

**{generate_prompts}**
```
{prompt_template}
```

**Topics should cover:**
"""
    for topic in conversation_topics:
        scenario_str += f"- {topic}\n"
    
    return scenario_str


def get_scenario_template_mapping() -> Dict[str, str]:
    return {scenario['name']: scenario['context'] for scenario in scenarios_json}


def validate_scenario(scenario: Dict) -> bool:
    required_fields = [
        "name", "context", "character_dynamic", 
        "conversation_topics", "characters"
    ]
    
    for field in required_fields:
        if field not in scenario:
            return False
            
    if not isinstance(scenario["conversation_topics"], list):
        return False
        
    if not isinstance(scenario["characters"], list):
        return False
        
    return True


def main():
    print("Testing scenario model utilities:")
    print("=" * 50)
    
    valid_scenarios = [s for s in scenarios_json if validate_scenario(s)]
    print(f"Valid scenarios: {len(valid_scenarios)}/{len(scenarios_json)}")
    
    template_map = get_scenario_template_mapping()
    print(f"Template mappings created: {len(template_map)}")
    
    if scenarios_json:
        test_scenario = scenarios_json[0]
        print(f"\nTesting json2prompt with scenario: {test_scenario['name']}")
        print(json2prompt(test_scenario))


if __name__ == "__main__":
    main()
