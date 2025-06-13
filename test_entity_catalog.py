import json

with open('../data/shared/entity_catalog.json', 'r', encoding='utf-8') as f:
    catalog = json.load(f)

# Explore structure
print(f"Total entities: {len(catalog['entities'])}")
print(f"Entities by level: {catalog['metadata']['entities_by_level']}")

# Find entities for a specific category
verkehr_entities = catalog['category_entity_map']['Verkehr']
print(f"Entities handling Verkehr issues: {verkehr_entities}")

# Get entity details
entity = catalog['entities']['BUND_BUNDESMINISTERIUM_DES_INNERN_UND_FÜR_HEIMAT']
print(f"Name: {entity['name']}")
print(f"Level: {entity['level']}")
print(f"Competencies: {entity['competencies']}")