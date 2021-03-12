import spacy
import json
# hotpotQA data format
# context: 10 documents; list of list (doc title, sentences (list of string))
# id: str
# answer: str
# question: str
# supporting_facts: List of list, (str, int) -> (doc title, sentence_idx)


def read_templates():
    template_file = "../template.csv"
    line_idx = 0
    rel2question = {}
    with open(template_file) as f:
        for line in f:
            line = line.strip()
            if line:
                line_idx += 1
                fields = line.split("\t")
                assert len(fields) == 5
                rel_code = fields[0]
                rel_template = fields[-1]
                assert rel_code not in rel2question
                rel2question[rel_code] = rel_template

    print(line_idx)
    print(len(rel2question))
    return rel2question

def get_question_answer(template, head, tail):
    answer = ""
    question = ""
    if "[HEAD]" in template:
        question = template.replace("[HEAD]", head)
        answer = tail
    elif "[TAIL]" in template:
        question = template.replace("[TAIL]", tail)
        answer = tail
    else:
        print("template wrong, no head neither tail")
        print(template)
    return question, answer


def split_to_two_docs(sents, evidences):
    evidences.sort()
    first_end = evidences[0] + 1
    first_doc = sents[:first_end]
    second_doc = sents[first_end:]
    return first_doc, second_doc


def docred2hotpot(rel2q, dr_file, miss_title, id_offset=0):
    # rel = "P118"
    #
    # dr_train = "/Users/xiaochen.hou1/git/DocRed/data/train_annotated.json"
    # dr_train = "/Users/xiaochen.hou1/git/DocRed/data/dev.json"
    train_data = json.load(open(dr_file))
    print(len(train_data))
    result_data = []
    one_supp_sents = []
    empty_supp_sents = 0
    multi_para = {}
    # DocRed data format
    # title
    # sents: list of list
    # vertex set: list of entities; name, sent_id, pos, type
    # labels: list of triples; head (id), tail (id), relation, evidence sentence id
    id = int(id_offset)
    print("start id ", id)
    for d in train_data:
        if d['title'] in miss_title:
            continue
        sents = []
        for sent in d['sents']:
            sent_text = " ".join(sent)
            sents.append(sent_text)
        context = [[d['title'], sents]]
        for label in d['labels']:
            head_ent = d['vertexSet'][label['h']][0]['name']
            tail_ent = d['vertexSet'][label['t']][0]['name']
            rel_code = label['r']
            question_template = rel2q[rel_code]
            question, answer = get_question_answer(question_template, head_ent, tail_ent)
            assert question != ""
            assert answer != ""

            converted_data = {}
            converted_data['_id'] = str(id)
            id += 1

            converted_data['answer'] = answer
            converted_data['question'] = question
            converted_data["level"] = "easy"
            converted_data["type"] = "bridge"

            evidences = label['evidence']
            support_facts = []

            if len(label['evidence']) == 0:
                empty_supp_sents += 1
                continue

            if len(label['evidence']) == 1:
                converted_data['context'] = context

                for e in evidences:
                    support_facts.append([d['title'], e])
                converted_data['supporting_facts'] = support_facts
                one_supp_sents.append(converted_data)

            else:
                first_doc, second_doc = split_to_two_docs(sents, evidences)
                converted_data['context'] = [[d['title'] + " 1", first_doc], [d['title'] + " 2", second_doc]]
                # converted_data['context'] = [[d['title'], first_doc], [d['title'], second_doc]]
                assert converted_data['_id'] not in multi_para
                multi_para[converted_data['_id']] = [[d['title']+ " 1", d['title']+ " 2"], [], []]
                sent_id_offset = len(first_doc)
                for e in evidences:
                    if e < sent_id_offset:
                        support_facts.append([d['title'] + " 1", e])
                        # support_facts.append([d['title'], e])
                    else:
                        support_facts.append([d['title'] + " 2", e-sent_id_offset])
                        # support_facts.append([d['title'], e - sent_id_offset])
                        assert  (e-sent_id_offset) >= 0
                converted_data['supporting_facts'] = support_facts
                result_data.append(converted_data)


            # print(d['vertexSet'][label['h']][0], d['vertexSet'][label['t']][0])
            # print("="*10)
    print("The size of empyt supporting sentences ", empty_supp_sents)
    print("The size of converted data ", len(result_data))
    print("end id ", id)
    # write_file = "converted_docred_train.json"
    # write_file = "converted_docred_dev.json"
    # json.dump(result_data, open(write_file,'w'))
    # print("The size of one supporting sentence data ", len(one_supp_sents))
    # one_supp_file = write_file + "_one_support.json"
    # # one_supp_file = "converted_docred_train_one_support.json"
    # json.dump(one_supp_sents, open(one_supp_file, 'w'))
    return result_data, one_supp_sents, multi_para, id


def check_error():
    error_file = "../error_log.txt"
    miss_title = set()
    with open(error_file) as f:
        for line in f:
            line = line.strip()
            if line:
                pos = line.find(" not exist in DB")
                title = line[:pos].strip()
                # print(title)
                miss_title.add(title)
    print(len(miss_title))
    print(miss_title)
    miss_title.add("TAE Technologies")
    return miss_title


if __name__ == "__main__":
    miss_title = check_error()
    rel2q = read_templates()
    #DocRed train file
    read_file = "/Users/xiaochen.hou1/git/DocRed/data/train_annotated.json"
    write_file = "converted_docred_total.json"
    result_data, one_supp_data, train_multi_para, end_id = docred2hotpot(rel2q, read_file, miss_title)
    # json.dump(result_data, open(write_file,'w'))
    # json.dump(one_supp_sents, open(one_supp_file, 'w'))
    #DocRed dev file
    read_file = "/Users/xiaochen.hou1/git/DocRed/data/dev.json"
    # write_file = "converted_docred_dev.json"
    dev_result_data, dev_one_supp_data, dev_multi_para, end_id = docred2hotpot(rel2q, read_file, miss_title, id_offset=end_id)
    result_data.extend(dev_result_data)
    one_supp_data.extend(dev_one_supp_data)
    print(len(result_data), len(one_supp_data))
    json.dump(result_data, open(write_file,'w'))
    json.dump(one_supp_data, open(write_file[:-5] +  "_one_support.json" , 'w'))

    for id in train_multi_para:
        assert id not in dev_multi_para
    total_multi_para = dict(train_multi_para)
    for id in dev_multi_para:
        total_multi_para[id] = dev_multi_para[id]
    assert len(total_multi_para) == len(result_data)
    multi_para_file = "multihop_para.json"
    json.dump(total_multi_para, open(multi_para_file, 'w'))

