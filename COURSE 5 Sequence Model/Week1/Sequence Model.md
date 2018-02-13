# Sequence Model

Tutor: Andrew Ng, deeplearning.ai

**Notation**:

- Superscript $[l]$ denotes an object associated with the $l^{th}$ layer. 
    - Example: $a^{[4]}$ is the $4^{th}$ layer activation. $W^{[5]}$ and $b^{[5]}$ are the $5^{th}$ layer parameters.

- Superscript $(i)$ denotes an object associated with the $i^{th}$ example. 
    - Example: $x^{(i)}$ is the $i^{th}$ training example input.

- Superscript $\langle t \rangle$ denotes an object at the $t^{th}$ time-step. 
    - Example: $x^{\langle t \rangle}$ is the input x at the $t^{th}$ time-step. $x^{(i)\langle t \rangle}$ is the input at the $t^{th}$ timestep of example $i$.

- Lowerscript $i$ denotes the $i^{th}$ entry of a vector.
    - Example: $a^{[l]}_i$ denotes the $i^{th}$ entry of the activations in layer $l$.

    ​

### Representing Word

1. 用一个真·字典来存储所有的单词（或者最经常出现的），其存储并编码排序所有的单词。
2. 用one hot方法来表示单词，结果是一个N维的vector（N为字典储存的所有的词数），其vector除第X项以外都是0。（X表示该单词在字典的位置）

如：Abandon是单词表第一个词，单词表有10000个词。那么代表着Abandon的向量用One hot来做的结果就是[1,0,0,0,0,0...,0]，除了第一个是1，其余都是0。

这么做的目的是通过以上过程得到的输入X，导入Sequence Model后得到一个目标输出y。

P.S: **如果遇到不在您的词汇表中的单词，该怎么办？ **那么答案就是，你创建了一个新的标记或者一个叫做Unknown Word的新的假字，注意如下，并且返回UNK表示不在你的词汇表中的单词

### RNN

#### 为什么不用传统神经网络

因为臣妾做不到啊！！！

1. 不同示例中的输入和输出可以是不同的长度。因此，并不是每个样例都具有相同的输入长度$T_x$或相同的输出长度$T_y$。也许每个句子都有最大长度，也许你可以填充或填充每个输入到最大长度，但这仍然不是一个好的表示。
2. 可能更严重的问题是，像这样一个朴素的神经网络架构，它**不会共享在技术人员不同位置上学到的features**。特别是，如果神经网络已经知道，可能出现在位置1的给出了一个符号，表明这是人名的一部分，那么如果它自动地发现在其他位置出现重物，$X_t$也会很好意味着这可能是一个人的名字。

### 1 - Forward propagation for the basic Recurrent Neural Network

Later this week, you will generate music using an RNN. The basic RNN that you will implement has the structure below. In this example, $T_x = T_y$. 

Here's how you can implement an RNN: 

**Steps**:
1. Implement the calculations needed for one time-step of the RNN.
2. Implement a loop over $T_x$ time-steps in order to process all the inputs, one at a time. 

## 1.1 - RNN cell

A Recurrent neural network can be seen as the repetition of a single cell. You are first going to implement the computations for a single time-step. The following figure describes the operations for a single time-step of an RNN cell. 

**Instructions**:
1. Compute the hidden state with tanh activation: $a^{\langle t \rangle} = \tanh(W_{aa} a^{\langle t-1 \rangle} + W_{ax} x^{\langle t \rangle} + b_a)$.
2. Using your new hidden state $a^{\langle t \rangle}$, compute the prediction $\hat{y}^{\langle t \rangle} = softmax(W_{ya} a^{\langle t \rangle} + b_y)$. We provided you a function: `softmax`.
3. Store $(a^{\langle t \rangle}, a^{\langle t-1 \rangle}, x^{\langle t \rangle}, parameters)$ in cache
4. Return $a^{\langle t \rangle}$ , $y^{\langle t \rangle}$ and cache

We will vectorize over $m$ examples. Thus, $x^{\langle t \rangle}$ will have dimension $(n_x,m)$, and $a^{\langle t \rangle}$ will have dimension $(n_a,m)$. 



