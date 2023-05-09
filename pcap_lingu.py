
import spacy
from spacy.tokens import Doc, Token
# from enum import Enum, auto
from functools import reduce
from typing import Optional
from nltk import Tree #type: ignore

from enum import Enum, auto
from typing import Optional

class proto(Enum):
    Ether = auto()
    Ip = auto()
    Ip6 = auto()
    Arp = auto()
    Rarp = auto()
    Tcp = auto()
    Udp = auto()
    Sctp = auto()
    Vlan = auto()
    Mpls = auto()
    Icmp = auto()
    Icmp6 = auto()
    
class typ(Enum):
    Host = auto()
    Net = auto()
    Port_type = auto()
    Portrange = auto()
    Proto = auto()
    Protochain = auto()

class direc(Enum):
    Src = auto()
    Dst = auto()
    Src_or_dst = auto()
    Src_and_dst = auto()

protos = [
    ("ether", proto.Ether),
    ("ethernet", proto.Ether),
    ("ip4", proto.Ip),
    ("ipv4", proto.Ip),
    ("IPV4", proto.Ip),
    ("IPv4", proto.Ip),
    ("ip6", proto.Ip6),
    ("ipv6", proto.Ip6),
    ("IPV6", proto.Ip6),
    ("IPv6", proto.Ip6),
    ("arp", proto.Arp),
    ("rarp", proto.Rarp),
    ("tcp", proto.Tcp),
    ("udp", proto.Udp),
    ("sctp", proto.Sctp),
    ("vlan", proto.Vlan),
    ("mpls", proto.Mpls),
    ("icmp", proto.Icmp),
    ("icmp6", proto.Icmp6)
]

typs = [
    ("host", typ.Host),
    ("hostname", typ.Host),
    ("net",typ.Net),
    ("network", typ.Net),
    ("port", typ.Port_type),
    ("portrange", typ.Portrange),
    ("proto", typ.Proto),
    ("protochain", typ.Protochain)
]

dirs = [
    ("src", direc.Src),
    ("source", direc.Src),
    ("dst", direc.Dst),
    ("destination", direc.Dst)
]




def dir_of_term(t: str) -> Optional[direc]:
    for s, d in dirs:
        if t == s:
            return d
    return None




def add_pcap_noun_rules(nlp):
    # Attribute ruler 
    ruler = nlp.get_pipe("attribute_ruler")
    
    # Add rules to the attribute ruler
    ruler.add( # type: ignore
        patterns = [[{"LOWER": txt}] for (txt, _) in protos], 
        attrs = {"TAG": "PROTO", "POS": "NOUN"}, 
        index=0
    )
    
    ruler.add( # type: ignore
        patterns = [[{"LOWER": txt}] for (txt, _) in typs], 
        attrs = {"TAG": "TYP", "POS": "NOUN"}, 
        index=0
    )
    
    ruler.add( # type: ignore
        patterns = [[{"LOWER": txt}] for (txt, _) in dirs], 
        attrs = {"TAG": "DIR", "POS": "NOUN"}, 
        index=0
    )
    
    ruler.add( # type: ignore
        patterns = [[{"LIKE_NUM": True}], [{"LIKE_URL": True}]], 
        attrs = {"TAG": "NN", "POS": "NOUN"}, 
        index=0
    )




###

# no negating, ever (because pcap '!' makes no sense)

# unidentifiable NOUN = foo (value)

###



nlp = spacy.load("en_core_web_sm")

# make all pcap field words nouns
add_pcap_noun_rules(nlp)




def print_doc(doc: Doc):
    print()
    print(doc)
    print()

def print_noun_chunks(doc: Doc):
    for chunk in doc.noun_chunks:
        print(
            list(chunk), " | ", 
            chunk.root.text, " | ",
            chunk.root.dep_, " | ",
            chunk.root.pos_, " | ",
            chunk.root.head.text
        )
        
    print()
    
def print_pos(doc: Doc):
    print([(w.text, w.pos_) for w in doc])
    print()

def print_tree_table(doc: Doc):
    i = 11
    j = 8
    print("text".ljust(i), 
            "dep".ljust(i), 
            "head.text".ljust(i), 
            "head.pos".ljust(i),
            "children")
    print("-" * 64)
    for token in doc:
        print(token.text.ljust(j), "| ", 
              token.dep_.ljust(j), "| ", 
              token.head.text.ljust(j), "| ", 
              token.head.pos_.ljust(j), "| ",
              [child for child in token.children])




def tok_format(tok: Token):
    return "_".join([tok.orth_, tok.tag_])


def to_nltk_tree(node: Token):
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
    else:
        return tok_format(node)
  
def print_tree(doc: Doc):
    [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]




def get_noun_root_chunks(doc: Doc) -> list[tuple[Token, list[Token]]]:
    
    def is_noun(n):
        return n.pos_ == "NOUN"
    def is_pronoun(n):
        return n.pos_ == "PRON"
    def is_determiner(n):
        return n.pos_ == "DET"
    
    def is_nsubjpass(n):
        return n.dep_ == "nsubjpass"
    def is_nsubj(n):
        return n.dep_ == "nsubj"
    
    all_nodes = [n for chunk in doc.noun_chunks for n in chunk]
    
    def head_last_noun_child(n):
        child = None
        for c in n.head.children:
            if (n != c and is_noun(c)):
                child = c
        return child
    
    def point_pronoun(n):
        return (n.head if is_nsubjpass(n) else 
                (head_last_noun_child(n) 
                 if (is_pronoun(n) and is_nsubj(n))
                 else n))
    
    def get_root_chunk(chunk):
        return (
            point_pronoun(chunk.root),
            [
                point_pronoun(n)
                for n in chunk 
                if (not is_determiner(n))
            ]
        )
    
    
    noun_root_chunks = [
        (r_ch)
        for chunk in doc.noun_chunks
        if ((r_ch := get_root_chunk(chunk)) and
            (is_noun(chunk.root) or 
             (is_pronoun(chunk.root) and
              ((r_ch[0]) not in all_nodes))))
    ]
    
    return noun_root_chunks
    
    
    

def print_noun_root_chunks(noun_root_chunks: list[tuple[Token, list[Token]]]):
    for root, chunk in noun_root_chunks:
        print(
            chunk, " | ", 
            root.text, " | ",
            root.dep_, " | ",
            root.pos_, " | ",
            root.head.text
        )
    
    
def remove_and_count_negations(noun_root_chunks: list[tuple[Token, list[Token]]]) -> tuple[list[tuple[Token, list[Token]]], int]:
    
    def is_negative(n):
        return n.text.lower() in ["without"]
    
    def is_not(n):
        return n.text.lower() in ["not", "n't"]
    
    def collect_nots(chunk):
        return set(() for n in chunk if is_not(n))
    
    
    # count all negations
    
    nots = [(collect_nots(root.head.children) 
             if (root != root.head) else set()) | 
            collect_nots(chunk) |
            ({root.head} if is_negative(root.head) else set())
            for root, chunk in noun_root_chunks]
    
    not_ct = len(reduce(lambda ns, rest: ns | rest, nots))
    
    # remove 'not's from inside chunk
    
    removed = [(root, [n for n in chunk if (not is_not(n))]) 
               for root, chunk in noun_root_chunks]
    
    return (removed, not_ct)
    
    
def is_conjunction(n):
    return n.pos_ == "CCONJ"
    
def is_src_or_is_dst(n):
    return dir_of_term(n.text) == direc.Src or dir_of_term(n.text) == direc.Dst
    
def is_direction_combo(chunk, i):
    return (
        (i-1 >= 0 and i+1 < len(chunk)) and
        is_conjunction(chunk[i]) and
        (is_src_or_is_dst(chunk[i-1]) and is_src_or_is_dst(chunk[i+1]))
    )
    
    
def is_proto(n):
    return n.tag_ == "PROTO"
def is_typ(n):
    return n.tag_ == "TYP"
def is_dir(n):
    return n.tag_ == "DIR"
def is_field_keyword(n):
        (is_proto(n) or is_typ(n) or is_dir(n))


def pcap_split_conjuctions(noun_root_chunks: list[tuple[Token, list[Token]]]) -> list[tuple[Token, list[Token]]]:
    
    def is_and(n):
        return (n.text == "and")
    
    def is_conj(n):
        return n.dep_ == "conj"
    
    
    
    def get_chunk_by_root(root, root_chunks) -> Optional[int]:
        for i, (r, _) in enumerate(root_chunks):
            if (r == root):
                return i
        return None
            
    def first_conjunction_child(n):
        for c in n.children:
            if (is_conjunction(c)):
                return c
    
    def split_chunk_by_conjunction(chunk, i = 0):
        
        if (len(chunk) == 0):
            return []
        elif (i >= len(chunk)):
            return [chunk]
        else:
            n = chunk[i]
            
            # DO NOT split after combo
            # return (
            #     ([chunk[:i-1]] if (i-1 > 0) else []) + 
            #     [chunk[i-1:i+2]] + 
            #     split_chunk_by_conjunction(chunk[i+2:], 0)
            # )
            
            if (is_and(n) and not is_direction_combo(chunk, i)):
                # split after conjunction (AND only)
                return (
                    [chunk[:i]] +
                    split_chunk_by_conjunction(chunk[i+1:], 0)
                )
            else:
                return split_chunk_by_conjunction(chunk, i+1)
    
    # flatten all conjunctions
    
    def flatten_conj(lst, i = 0):
        if (i >= len(lst)):
            return lst
        else:
            root, chunk = lst[i]
            
            if is_conj(root):
                h_i = get_chunk_by_root(root.head, lst)
                _, head_chunk = lst[h_i]
                joined_chunk = (
                    head_chunk + 
                    [first_conjunction_child(root.head)] + 
                    chunk
                )
                
                lst.pop(max(i, h_i))
                lst.pop(min(i, h_i))
                
                lst.insert(min(i, h_i), (root, joined_chunk))
                
            return flatten_conj(lst, i + 1)
    
    flats = flatten_conj(noun_root_chunks)
    
            
    # split each chunk into a list of chunks and then concat (and add roots)
    
    def concat(lst):
        return [x for l in lst for x in l]
    
    return [
        (chunk[-1], chunk)
        for chunk in concat([
            split_chunk_by_conjunction(flat)
            for _, flat in flats
        ])
    ]
    
    
    

    
def print_relationship(rel: tuple[Optional[Token], Optional[Token], Optional[Token]]):
    print(rel)







# language theory:
#
# NOT is ambiguous in both English and pcap expressions
#
# is this a bug specific to pcap 
#   or a feature of programming languages?


sentences = [
    # MAIN sentence
    "ip4 that has a source host of foo",
    # simple alternative sentences
    "ip4 with a source host of foo",
    "ip4 with source host foo",
    "ip4 with foo as the source host",
    "ip4 where the source host is foo",
    
    # NOTS
    
    # MAIN sentence
    "ip4 that does not have a source host of foo",
    "ip4 that doesn't have a source host of foo",
    "not ip4 that has a source host of foo",
    # simple alternative sentences
    "ip4 without a source host of foo",
    "ip4 not with source host foo",
    "ip4 without foo as the source host",
    "ip4 where the source host is not foo",
]

sentences_with_repetition = [
    # "host host"
    "ip4 that has a host of host",
    
    # NOTS
    
    # "host host"
    "ip4 that does not have a host of host",
]

sentences_with_proto = [
    
    # FIXME
    
    # make a get_chunk_by_head() 
    # and find the sibling of each protocol
    # to see whether proto is referring to ip4  or icmp6
    
    "ip4 that has a proto of icmp",
    
    # UH OH:
    #
    # language theory:
    #
    # protocol breaks spacy's English AST 
    
    
    # UH OH ambiguous sentiment
    "a packet with ip4 protocol and protocol icmp",
    
    "ether with protocol tcp and protocol ip4",
    "ether with protocol tcp and ip4 protocol",
]

advanced_sentences = [
    # redundancy
    "ip4 with source host foo as the source host",
    # ambiguous anaphora
    "it is ip4 with a source host of foo",
    # reference to packet
    "the packet is ip4 with a source host of foo",
    # ambiguous anaphora + reference to packet
    "it is an ip4 packet with a source host of foo",
    
    # NOTS
    
    # redundancy
    "ip4 without source host foo as the source host",
    # ambiguous anaphora
    "it is not ip4 with a source host of foo",
    "it is ip4 without a source host of foo",
    # reference to packet
    "the packet is not ip4 with a source host of foo",
    "the packet is ip4 without a source host of foo",
    # ambiguous anaphora + reference to packet
    "it is not an ip4 packet with a source host of foo",
    "it is an ip4 packet without a source host of foo",
    
]


sentences_with_conjs = [
    # conjunctions for clarity
    "ip4 that has direction source and type host and value foo",
    
    # src_or_dst / src_and_dst
    "ip4 that has source or destination host foo",
    "ip4 that has source and destination host foo",
    
    # NOTS
    
    # conjunctions for clarity
    "ip4 that does not have direction source and type host and value foo",
    
    # src_or_dst / src_and_dst
    "ip4 that does not have source or destination host foo",
    "ip4 that does not have source and destination host foo",
]


for txt in sentences + sentences_with_conjs + sentences_with_proto:
    doc = nlp(txt)
    print_doc(doc)
    # print_noun_chunks(doc)
    # print_noun_root_chunks(get_noun_root_chunks(doc))
    # print()
    n_cks, ct = remove_and_count_negations(get_noun_root_chunks(doc))
    print(ct)
    # print_noun_root_chunks(n_cks)
    print()
    print_noun_root_chunks(pcap_split_conjuctions(n_cks))
    
    # print_pos(doc)
    # print_tree_table(doc)
    print_tree(doc)
    
    




























