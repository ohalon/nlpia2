>>> ! ruby scripts/extract_natural_lang.rb
>>> ch7 = yaml.full_load(open('/home/hobs/code/tangibleai/nlpia-manuscript/manuscript/adoc/Chapter 07 -- Getting Words in Order with Convolutional Neural Networks (CNNs).adoc.yml'))
>>> ch7[3]
{'first-sentence': '_Convolutional Neural Networks_ (CNNs) are all the rage for _computer vision_ (image processing).\n',
 'last-sentence': "And there's not a single reference implementation example in the PyTorch or Keras packages themselves!\n"}
>>> ch7[0]
{'block_type': 'section',
 'level': 1,
 'section_title': 'Finding patterns in words with convolutional neural networks (CNNs)'}
>>> ch7[1]
{'first-sentence': 'This chapter covers\n', 'last-sentence': ''}
>>> ch7[2]
{'first-sentence': 'In this chapter you will unlock the misunderstood superpowers of convolution for Natural Language Processing.\n',
 'last-sentence': ''}
>>> ch7[3]
{'first-sentence': '_Convolutional Neural Networks_ (CNNs) are all the rage for _computer vision_ (image processing).\n',
 'last-sentence': "And there's not a single reference implementation example in the PyTorch or Keras packages themselves!\n"}
>>> hist ~0/-10--1 -o -p
>>> hist ~0/-10-
>>> hist ~0/10-
>>> hist ~0/
>>> hist ~0/-10
>>> hist ~0/-10-20
