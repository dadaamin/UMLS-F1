def get_UMLS_entities(doc): 
    global nlp, linker
    if nlp is None:
        import spacy
        import scispacy
        from scispacy.linking import EntityLinker
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        print("Loading scispacy UMLS Linker...")
        linker = nlp.get_pipe("scispacy_linker")

    entities = set()
    for entity in doc.ents:
        if entity._.kb_ents:
            entities.add(linker.kb.cui_to_entity[entity._.kb_ents[0][0]].canonical_name)
    return entities

def compute_UMLS_F1(model_output, label):
    global nlp, linker
    if nlp is None:
        import spacy
        import scispacy
        from scispacy.linking import EntityLinker
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        print("Loading scispacy UMLS Linker...")
        linker = nlp.get_pipe("scispacy_linker")
    
    
    doc_model_output = nlp(model_output)
    doc_label = nlp(label)

    model_output_entities = get_UMLS_entities(doc_model_output)
    label_entities = get_UMLS_entities(doc_label)

    if len(model_output_entities) == 0:
        P = 0.0
    else:
        P = len([pred for pred in model_output_entities if pred in label_entities]) / len(model_output_entities)
    if len(label_entities) == 0:
        R = 0.0
    else:
        R = len([l for l in label_entities if l in model_output_entities]) / len(label_entities)

    if (P + R) == 0:
        F = 0.0
    else:
        F = 2 * P * R / (P + R)

    return P, R, F
