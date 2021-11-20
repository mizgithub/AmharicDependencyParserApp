from typing import Counter
import streamlit as st


#**************** POS tagger*********************************************
def posTagModel(input_dim = 9160, hidden_neurons = 128, output_dim = 23):
    model = Sequential([
        Dense(hidden_neurons, input_dim = input_dim),
        Activation('relu'),
        Dropout(0.2),
        Dense(hidden_neurons),
        Activation('relu'),
        Dropout(0.2),
        Dense(output_dim, activation = 'softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer = 'Adam', metrics = ['acc'])
    return model
def add_basic_features(sentence_terms, index):
    term = sentence_terms[index]
    return{
        'nb_terms': len(sentence_terms),
        'term': term,
        'is_first': index == 0,
        'is_last': index == len(sentence_terms) - 1,
        'prefix-1': term[0],
        'prefix-2': term[:2],
        'prefix-3': term[:3],
        'suffix-1': term[-1],
        'suffix-2': term[-2:],
        'suffix-3': term[-3:],
        'prev_word': '' if index == 0 else sentence_terms[index - 1],
        'next_word': '' if index == len(sentence_terms) - 1 else sentence_terms[index + 1]
    }
def transform_to_Predictable_dataset(sent):
    x=[]
    for index, w in enumerate(sent):
        x.append(add_basic_features(sent, index))
    return x
def POS_tag(sentence):
    #loading model, dict_vectorizer and label_encoder
    model = posTagModel()
    model.load_weights("./POS_tag_module/tagClassifier2.h5")

    vect_file = open("./POS_tag_module//dict_vectorizer.pkl", "rb")
    dic_vectorizer = pickle.load(vect_file)

    encode_file = open("./POS_tag_module//label_encoder.pkl", "rb")
    label_encoder = pickle.load(encode_file)
    sentence = ' '.join(sentence.split())
    sent_array = []
    for i in sentence.split(" "):
        sent_array.append(i)
    featured_sentence = transform_to_Predictable_dataset(sent_array)
    transformed_sentence = dic_vectorizer.transform(featured_sentence)
    POS_tag_predicted = model.predict(transformed_sentence)
    final_post_tag_result = []
    for i in POS_tag_predicted:
        final_post_tag_result.append(np.argmax(i))
    
    encoded_pos_tag_result = label_encoder.inverse_transform(final_post_tag_result)
    
    # final_result = []
    # for i in range(0, len(sent_array)):
    #     final_result.append((sent_array[i], encoded_pos_tag_result[i]))
        
    return sent_array, encoded_pos_tag_result
#****************Parser models*********************
def parserModel(word2index, tag2index,maxlen):
    ## for predicting the four transition actions
    stack_input=Input(shape=(maxlen,))
    stack_encoder=Embedding(len(word2index), 128, trainable=True)(stack_input)
    stack_pos=Input(shape=(maxlen,))
    stack_pos_encoder=Embedding(len(tag2index), 128, trainable=True)(stack_pos)
    buffer_input=Input(shape=(maxlen,))
    buffer_encoder=Embedding(len(word2index), 128, trainable=True)(buffer_input)
    buffer_pos=Input(shape=(maxlen,))
    buffer_pos_encoder=Embedding(len(tag2index),128, trainable=True)(buffer_pos)
    match1= dot([stack_encoder, stack_pos_encoder],  axes=[2,2])
    stack_word_pos=LSTM(256,return_sequences=True)(match1)
    match2=dot([buffer_encoder, buffer_pos_encoder], axes=[2,2])
    buffer_word_pos=LSTM(256, return_sequences=True)(match2)
    output=concatenate([stack_word_pos, buffer_word_pos],axis=-1)
    output=LSTM(256, return_sequences=True)(output)
    output=SeqSelfAttention(128,attention_activation='sigmoid')(output)
    output=TimeDistributed(Dense(4))(output)
    output=Activation('softmax')(output)
    model=Model(inputs=[stack_input, stack_pos, buffer_input, buffer_pos], outputs=output)
    #model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['acc',self.ignore_class_accuracy(0)])
    return model
def parserRelModel(word2index, tag2index,maxlen):
    ## for predicting relationship types
    wordPos=Input(shape=(maxlen,))
    wordpos_embed=Embedding(len(tag2index), 256, trainable=True)(wordPos)
    headPos=Input(shape=(maxlen,))
    headPos_embed=Embedding(len(tag2index), 256, trainable=True)(headPos)        
    match1= concatenate([wordpos_embed, headPos_embed],  axis=-1)
    output=Bidirectional(LSTM(256, return_sequences=True))(match1)
    output=Bidirectional(LSTM(256, return_sequences=True))(output)
    output=TimeDistributed(Dense(36))(output)
    output=Activation('softmax')(output)
    model=Model(inputs=[wordPos, headPos], outputs=output)
    #model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['acc',self.ignore_class_accuracy(0)])
    return model

def getWordLookUp():
    data=""
    with open("./dependencyparser_module//word2index.json",'r', encoding="utf-8") as fp:
        data=json.load(fp)
    return(data)
def getPOSLookUp():
    data=""
    with open("./dependencyparser_module//tag2index.json",'r', encoding="utf-8") as fp:
        data=json.load(fp)
    return(data)
def getRelTypelookUp():
    data=""
    with open("./dependencyparser_module//rel2index.json", "r", encoding='utf-8') as fp:
        data=json.load(fp)
    return data
def vectorize(buffer, buffer_pos):
        
    ## converting the words into integer using word2index
    buffer_state=[]
    buffer_pos_state=[]
    
    ## loading word2index
    word2index= getWordLookUp()
    tag2index=getPOSLookUp()
    stack=[word2index['-root-']] ## the first element in the stack root
    stack_pos=[tag2index['-root-']] ## the first element in the stack POS root
    for w in buffer:
        try:
            buffer_state.append(word2index[w])
        except KeyError:
            buffer_state.append(word2index['-OOV-'])
    for t in buffer_pos:
        try:
            buffer_pos_state.append(tag2index[t])
        except KeyError:
            buffer_pos_state.append(tag2index['-OOV-'])
    ## pading zero
    maxlen=25
    for i in range(1, maxlen):
        stack.append(0)
    for i in range(1, maxlen):
        stack_pos.append(0)
    for i in range (len(buffer_state), maxlen):
        buffer_state.append(0)
    for i in range(len(buffer_pos_state), maxlen):
        buffer_pos_state.append(0)
    ## reshaping the inputs in to (1,25)
    stack=np.reshape(np.array(stack), (1,maxlen))
    stack_pos=np.reshape(np.array(stack_pos), (1,maxlen))
    buffer_state=np.reshape(np.array(buffer_state),(1,maxlen))
    buffer_pos_state=np.reshape(np.array(buffer_pos_state),(1,maxlen))
    return stack, stack_pos, buffer_state, buffer_pos_state, buffer,buffer_pos, word2index,tag2index, maxlen
  
def getTransition(buffer,pos):
    stack, stack_pos, buffer_state, buffer_pos_state, buffer,buffer_pos, word2index, tag2index, maxlen=vectorize(buffer, pos)
    stack_top=0
    buffer_top=0
    tran=[]
    ## getting the trained model 
    model=parserModel(word2index, tag2index, maxlen)
    model.load_weights("./dependencyparser_module//TBNADP.h5")
    while buffer_state[0][buffer_top]!=0:
        #print(".", end='')
        transition=model.predict([stack, stack_pos, buffer_state, buffer_pos_state])
        y=transition.argmax(axis=2)
        y=y[0][22]
        tr_name=""
        if y==0 or y==2:
            #shift
            if y==0:
                tr_name="SHIFT"
            else:
                tr_name="RIGHT-ARC"
            stack_top=stack_top+1
            stack[0][stack_top]=buffer_state[0][buffer_top]
            stack_pos[0][stack_top]=buffer_pos_state[0][buffer_top]
            buffer_state=np.delete(buffer_state,buffer_top,1)
            buffer_state=np.append(buffer_state,0)
            buffer_state=np.array(buffer_state)
            buffer_state=np.reshape(buffer_state,(1,maxlen))
            buffer_pos_state=np.delete(buffer_pos_state,buffer_top,1)
            buffer_pos_state=np.append(buffer_pos_state,0)
            buffer_pos_state=np.array(buffer_pos_state)
            buffer_pos_state=np.reshape(buffer_pos_state,(1,maxlen))
            tran.append(y)
        elif y==1 or y==3:
            if y==1:
                tr_name="LEFT-ARC"
            else:
                tr_name="REDUCE"
            #left arc
            stack[0][stack_top]=0
            stack_pos[0][stack_top]=0
            stack_top=stack_top-1
            tran.append(y)
        #print(y,"-------->",tr_name)
    return buffer,buffer_pos,tag2index,tran
def unlabledParse(buffer,pos):
        
    buffer, _, _, tran=getTransition(buffer,pos)
    s_stack=[0]
    s_index1=[]
    heads=[]
    c=1
    for i in buffer:
        s_index1.append(c)
        c=c+1
        heads.append(0)
    #s_index1
    heads.append(0)
    b=0
    s=0
    for i in tran:
        if i==0:
            s_stack.append(s_index1[b])
            b=b+1
            s=s+1
        elif i==1:
            heads[s_stack[s]]=s_index1[b]
            s_stack.pop()
            s=s-1
            
        elif i==2:
            heads[s_index1[b]]=s_stack[s]
            s_stack.append(s_index1[b])
            s+=1
            b+=1
            
        else:
            s_stack.pop()
            s-=1
    heads.pop(0)
    unlabled_tree = []
    for i in range(len(buffer)):
        row  = str(i+1)+"\t"+buffer[i]+"\t"+pos[i]+"\t"+str(heads[i])
        unlabled_tree.append(row)
    
    return unlabled_tree
def constructrelation(unlabledTree):
    wordp,headp = [],[]
    for row in unlabledTree:
        index,w,p,hi = row.split("\t")
        if int(hi)!=0:
            hi=int(hi)-1  # index start from zero
            _,_,hp,_ = unlabledTree[hi].split("\t")
        else:
            hp = 'root'
        wordp.append(p)
        headp.append(hp)
    tag2index=getPOSLookUp()
    rel2index=getRelTypelookUp() 
    word2index=getWordLookUp()
    
    ## converting texts in to number value
    wordPSet, headPSet = [],[]
    
    ## word pos
    
    for wr in wordp:
        try:
            wordPSet.append(tag2index[wr])
        except KeyError:
            wordPSet.append(tag2index['-OOV-'])
    ## head word pos
    wordPSet = np.array(wordPSet)
    wordPSet = wordPSet.reshape(1, len(wordPSet))
    for wr in headp:
       
        try:
            headPSet.append(tag2index[wr])
        except KeyError:
            headPSet.append(tag2index['-OOV-'])
    headPSet = np.array(headPSet)
    headPSet = headPSet.reshape(1, len(headPSet))
    mxlen=25
    
    wordPSet=pad_sequences(wordPSet, maxlen=mxlen, dtype='int32', padding='post', truncating='post')
    headPSet=pad_sequences(headPSet, maxlen=mxlen, dtype='int32', padding='post', truncating='post')
    
    model=parserRelModel(word2index, tag2index,mxlen)
    model.load_weights("./dependencyparser_module//TBNADPrel.h5")
    y=model.predict([wordPSet,headPSet])
    y = np.array(y)
    y = y.reshape(y.shape[0], y.shape[1], y.shape[2])
    
    labeledTree = []
    i=0
    
    for srel in y:
        for rel in srel:
            i+=1
            if i>len(unlabledTree):
                break
            #rel = srel[len(unlabledTree)-1]
            _,wrd, pos,hd=unlabledTree[i-1].split("\t")
            rtxt=list(rel2index.keys())[list(rel2index.values()).index(rel.argmax())]
            row=str(i)+"\t"+wrd+"\t"+pos+"\t"+hd+"\t"+rtxt
            labeledTree.append(row)
            
    return labeledTree

def singleSentenceParser(sentence, type):
    buffer, POStag = POS_tag(sentence)

    ulabledDependencyTree = unlabledParse(buffer, POStag)
    if type=="Unlabeled":
        return ulabledDependencyTree
    else:
        return constructrelation(ulabledDependencyTree)

def singleSentenceParsingPage():
    form = st.form(key = "input form")
    sentence = form.text_input("Enter Amharic sentence")
    parseOption  = form.radio("", ["Unlabeled", "Labeled"])
    if form.form_submit_button("Parse"):
        with st.spinner(text = "Please wait..."):
            result = singleSentenceParser(sentence, parseOption)
        st.success("Result")
        if parseOption == "Unlabeled":
            htmlString = "<table><tr><th>Index</th><th>Word</th><th>Word POS tag</th><th>Head-word index</th></tr>"
            for r in result:
                i,w,wp,hi = r.split("\t")
                htmlString+="<tr><td>"+i+"</td><td>"+w+"</td><td>"+wp+"</td><td>"+hi+"</td></tr>"
            htmlString+="</table>"
            # st.markdown(htmlString,unsafe_allow_html = True)
            
            graph = graphviz.Digraph(format="png")
            
            graph.node('ROOT')
            for i in range(len(result)):
                _,w,wp,hi = result[i].split("\t")
                graph.node(w+"["+wp+"]")

            
            for i in range(len(result)):
                _,w,wp,hi = result[i].split("\t")
                hi = int(hi)
                if hi != 0:
                    hi = hi-1
                    _,hw,hp,_ = result[hi].split("\t")
                    hw = hw+"["+hp+"]"
                else:
                    hw = "ROOT"
                graph.edge(w+"["+wp+"]", hw)
            #st.graphviz_chart(graph)

        else:
            htmlString = "<table><tr><th>Index</th><th>Word</th><th>Word POS tag</th><th>Head-word index</th><th>Dependency relation</th></tr>"
            for r in result:
                i,w,wp,hi,dr = r.split("\t")
                htmlString+="<tr><td>"+i+"</td><td>"+w+"</td><td>"+wp+"</td><td>"+hi+"</td><td>"+dr+"</td></tr>"
            htmlString+="</table>"
            # st.markdown(htmlString,unsafe_allow_html = True)
            # st.markdown("<hr>", unsafe_allow_html = True)
            graph = graphviz.Digraph()
            
            
            graph.node("ROOT")
            for i in range(len(result)):
                _,w,wp,hi,_ = result[i].split("\t")
                graph.node(w+"["+wp+"]")

            
            for i in range(len(result)):
                _,w,wp,hi,dr = result[i].split("\t")
                hi = int(hi)
                if hi != 0:
                    hi = hi-1
                    _,hw,hp,_,_ = result[hi].split("\t")
                    hw = hw+"["+hp+"]"
                else:
                    hw = "ROOT"
                graph.edge(w+"["+wp+"]", hw, label=dr)
            #st.graphviz_chart(graph)

        
        
        st.markdown(htmlString, unsafe_allow_html = True)
        st.markdown("<hr>", unsafe_allow_html = True)
        st.subheader("Graph Representation")
        st.graphviz_chart(graph)
    
        st.graphviz_chart('''
            digraph {
                run -> intr
                intr -> runbl
                runbl -> run
                run -> kernel
                kernel -> zombie
                kernel -> sleep
                kernel -> runmem
                sleep -> swap
                swap -> runswap
                runswap -> new
                runswap -> runmem
                new -> runmem
                sleep -> runmem
            }
        ''')

        

st.header("Amharic Dependency Parser System")
st.markdown("<hr>", unsafe_allow_html=True)

with st.spinner("Loading modules ..."):
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.preprocessing import LabelEncoder
    from keras.models import Sequential
    import pickle
    import json
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import Sequential, load_model
    from keras.layers import Dense, LSTM, Activation, TimeDistributed, Embedding, Input, InputLayer, Dropout,Bidirectional
    from tensorflow.keras.optimizers import Adam,SGD
    from keras.regularizers import l2
    from keras.layers import concatenate, add, dot
    from keras import Model
    from keras_self_attention import SeqSelfAttention
    import graphviz as graphviz
    import matplotlib.pyplot as plt

option = st.sidebar.radio("", ["Single sentence parser", "Batch parser"])
if option == "Single sentence parser":
    singleSentenceParsingPage();

st.markdown("<hr><br><br><br><br><br><center><p>Developed by Mizanu Zelalem<br><i>Faculty of computing and Informatics, Jimma Institute of Technology, Jimma university, Jimma, Ethiopia</i><br><i>mizanu143@gmail.com</i></p></center>", unsafe_allow_html = True)