# PyTorch-Batch-Seq2seq

This is a for batched sequence-to-sequence (seq2seq) models implemented in PyTorch modified from offical pytorch tutorial.
It uses batched GRU encoder and GRU decoder(no attention).
This code presents task on string reverse and initialize embedding layer with one-hot encoding. It works well and achieves about 70% accuracy after 200 epochs. 

Usage: 
 - python seq2seq_translation_tutorial.py

Please refer to offical pytorch tutorial on "Translation with a Sequence to Sequence Network and Attention"<br>
PyTorch version mechanism illustration, see here: <br>
http://pytorch.org/tutorials/_images/decoder-network.png<br>
PyTorch offical Seq2seq machine translation tutorial:<br>
http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html<br>
<br>

