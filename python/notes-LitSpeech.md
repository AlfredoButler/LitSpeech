




## Beam search 

- **encoder**: A model that represents sequences as vectors. Each sequence is encoded into a vector representation.  
- **decoder**: A model that is used to generate the previous and subsequent sequences. Usually a recurrent neural network.

Both the encoder and decoder are trained simultaneously, and the trained encoder can be used to compute a distributed representation of any sequence. 


A greedy decoder works, but it isn't the most accurate. 
- **"greedy" transition-based parsing**: A type of parsing where the parser tries to make the best decision at each configuration. This can lead to search errors when an early decision locks the parser into a poor derivation. 
- **greedy search decoder**: A decoder in which the generation of some output sequence is a result of selecting the most likely prediction at each step. 
> "[The Greedy search decoder] approach has the benefit that it is fast, but the quality of the final output sequences may be far from optimal." - Brownlee


approximate search/decoding algorithms such as greedy decoding or beam search. 

# References
- Brownlee, J. 2020. *How to Implement a Beam Search Decoder for Natural Language Processing*. https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
- 