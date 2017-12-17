#_*_coding:utf-8_*_
#deprecated
def get_tags2(line):
    b = line.split('\n')
    print b
    d = []
    for i in b:
        print i
        if i != '':
            tmp = ''
            for j in range(len(i)):
                if (i[j]>= 'A' and i[j] <= 'Z'):
                    tmp += i[j]
                else:
                    if tmp != '':
                        d.append(tmp)
                        break
                    else:
                        break

def get_tags(line):
    a = line.split('\n')
    print a
    lst = []
    for i in a:
        if i == '':
            continue
        beg = i.find('(')
        pos = i.find(' ', beg)
        print pos
        if pos == -1:
            t = i[beg+1:]
        else:
            t = i[beg+1: pos]
        lst.append(t)
    for i in lst:
        if not i.isalpha():
            lst.remove(i)
    return lst

if __name__ == '__main__':
    line = '''(ROOT
  (S
    (NP (FW i))
    (VP (VBP 'm)
      (NP
        (NP (DT the) (NN monkey))
        (PP (IN in)
          (NP
            (NP (DT the) (JJ white) (NN shirtplease))
            (SBAR
              (S
                (S
                  (VP (VB leave)
                    (NP (DT a) (JJ like) (NN comment))))
                (CC and)
                (S
                  (INTJ (VB please))
                  (VP (VB subscribe)))))))))
    (. !)))
'''
    print get_tags(line)
    abb_word = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
    abb_phrase = ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X']
    abb = abb_word
    abb.extend(abb_phrase)
    nlp = StanfordCoreNLP('http://localhost:9000')

