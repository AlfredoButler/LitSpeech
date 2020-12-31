# Beam search 

Core questions:
- What is beam search?
- What is greedy search?

## Encoder-decoder model

- **encoder**: A model that represents sequences as vectors. Each sequence is encoded into a vector representation.  
- **decoder**: A model that takes in encoded inputs and uses them to generate the output sequences. Usually a recurrent neural network.

Both the encoder and decoder are trained simultaneously, and the trained encoder can be used to compute a distributed representation of any sequence. 

## Seq2seq (2014)

Seq2Seq, or Sequence To Sequence, is a model used in sequence prediction tasks, such as language modeling and machine translation. 
1. One LSTM is used as an encoder to read each input sequence one timestep at a time, obtaining a fixed dimensional vector representation (called a context vector)
2. Another LSTM is used as the decoder to extract the output sequence from the context vector. This decoder is conditioned on the input sequence. 
 
Seq2seq is an application of the encoder-decoder model. 

## Greedy decoder

A greedy decoder works, but it isn't the most accurate. 
- **"greedy" transition-based parsing**: A type of parsing where the parser tries to make the best decision at each configuration. This can lead to search errors when an early decision locks the parser into a poor derivation. 
- **greedy search decoder**: A decoder in which the generation of some output sequence is a result of selecting the most likely prediction at each step. 
> "[The Greedy search decoder] approach has the benefit that it is fast, but the quality of the final output sequences may be far from optimal." - Brownlee

## Beam Search

To explain beam search, we'll consider an example. Suppose we have the use the sequence to sequence use case.



## Connectionist Temporal Classification (CTC)







---

# References 

- Sequence to Sequence Learning with NNs [[article]](https://paperswithcode.com/method/seq2seq) [[paper]](https://arxiv.org/pdf/1409.3215v3.pdf)

- Khandelwal, R. (2020). *An intuitive explanation of Beam Search*. https://towardsdatascience.com/an-intuitive-explanation-of-beam-search-9b1d744e7a0f

- Brownlee, J. (2020). *How to Implement a Beam Search Decoder for Natural Language Processing*. https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/

- Hannun, A. (2017). Sequence modeling with ctc. *Distill*, 2(11), e8. https://distill.pub/2017/ctc/
