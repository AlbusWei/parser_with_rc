import random


def power_set(set1):
    ans = [[]]
    for i in set1:
        l = len(ans)
        for j in range(l):
            t = []
            t.extend(ans[j])
            t.append(i)
            ans.append(t)
    return ans


def divide_list(source_list, num_of_slices):
    # 返回词组的一个分划
    ret_list = []
    cut_point = []
    for x in range(num_of_slices):
        cut_point.append(random.randint(0, len(source_list)))
    cut_point = sorted(cut_point)
    ptr = 0
    for x in range(num_of_slices):
        ret_list.append(source_list[ptr:cut_point[x]])
        ptr = cut_point[x]
    return ret_list


def generate_sentences_samples(word_dict, size=500, output_file="tests.txt"):
    output_list = []
    det_list = []
    adj_list = []
    noun_list = []
    verb_list = []
    adv_list = []

    for key, value in word_dict.items():
        if "adjective" == value["type"]:
            adj_list.append(key)
        elif "noun" == value["type"]:
            noun_list.append(key)
        elif "determinant" == value["type"]:
            det_list.append(key)
        elif "adverb" == value["type"]:
            adv_list.append(key)
        elif "trans_verb" == value["type"]:
            verb_list.append(key)
    # 生成副词的幂集
    pow_adv = power_set(adv_list)

    # 生成4 * size个句子
    for x in range(4 * size):
        # 从名词列表中随机抽取total_noun个名词，用作主语的有sub_noun个，其余obj_noun个作为宾语
        random.shuffle(noun_list)
        total_noun = random.randint(2, len(noun_list) - 1)
        sub_noun = random.randint(1, total_noun - 1)
        obj_noun = total_noun - sub_noun

        # 从形容词列表中随机抽取total_adj个形容词，修饰主语的有sub_adj个，其余obj_adj个修饰宾语
        random.shuffle(adj_list)
        total_adj = random.randint(0, len(adj_list))
        sub_adj = random.randint(0, total_adj)
        obj_adj = total_adj - sub_adj

        a_flag = True
        the_flag = True

        # 拼凑主语部分
        sub_arranged_adj = divide_list(adj_list[:sub_adj], sub_noun)
        sub_component = ""
        for y in range(sub_noun):
            if a_flag and random.random() < 0.025:
                sub_component += 'a '
                a_flag = False
            elif the_flag and random.random() < 0.025:
                sub_component += 'the '
                the_flag = False
            for z in range(len(sub_arranged_adj[y])):
                sub_component += sub_arranged_adj[y][z]
                sub_component += ' '
            sub_component += noun_list[y]
            if y < sub_noun - 1:
                sub_component += ' '

        # 拼凑宾语部分
        obj_arranged_adj = divide_list(
            adj_list[sub_adj:sub_adj + obj_adj], obj_noun)
        obj_component = ""

        for y in range(obj_noun):
            if a_flag and random.random() < 0.025:
                obj_component += 'a '
                a_flag = False
            elif the_flag and random.random() < 0.025:
                obj_component += 'the '
                the_flag = False
            for z in range(len(obj_arranged_adj[y])):
                obj_component += obj_arranged_adj[y][z]
                obj_component += ' '
            obj_component += noun_list[sub_noun + y]
            if y < obj_noun - 1:
                obj_component += ' '

        # 拼凑谓语部分
        adv_component = ""
        chosen_adv = pow_adv[random.randint(0, len(pow_adv) - 1)]

        for y in range(len(chosen_adv)):
            adv_component += chosen_adv[y]
            if y < len(chosen_adv) - 1:
                adv_component += ' '

        verb_component = ""
        verb_component += verb_list[random.randint(0, len(verb_list) - 1)]
        verb_component += ' '
        verb_component += adv_component

        output_list.append(sub_component + ' ' +
                           verb_component + ' ' + obj_component)

    random.shuffle(output_list)
    output_list = output_list[:size]
    with open(output_file, "w") as f:
        for line in output_list:
            f.write(line + '\n')
