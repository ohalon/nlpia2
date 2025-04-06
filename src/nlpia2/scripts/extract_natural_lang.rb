#!/usr/bin/env ruby
# ARGV.each do|a|
#   puts "Argument: #{a}"
# end
require 'asciidoctor'
require 'yaml'

chapter_path = 'manuscript/adoc/Chapter 07 -- Getting Words in Order with Convolutional Neural Networks (CNNs).adoc'    # replace 

if ARGV.length == 1
  # chapter_path, *unused_argv = ARGV
  chapter_path = ARGV.at(0)
else
  puts "No ARGV found."
end

# This Ruby script extracts the "essence" of an adoc file, that is...
#
# 1. the headings 
# 2. the first sentence ("topic sentence") and last sentence ("punchline") of each paragraph
# 3. attempts to extract minimal descriptive info of other content like admonitions and listings - IN PROGRESS
#
# It helps authors with large adoc files determine whether each paragraph serves each section's purpose, 
# and whether each section builds up to the purpose of the adoc.
#
# It relies on a common tenet of good writing, that the first sentence of a paragraph, the "topic sentence", 
# sets the thesis for the paragraph and that the final sentence of a paragraph, if different, 
# constitutes the punchline, as if the paragraph is a mini-essay.
# 
# How I use it (in iPython):
#
# ```python
# >>> ! ruby scripts/extract_natural_lang.rb
# >>> ch7 = yaml.full_load(open('/home/hobs/code/tangibleai/nlpia-manuscript/manuscr
# ipt/adoc/Chapter 07 -- Getting Words in Order with Convolutional Neural Networks (
# CNNs).adoc.yml'))
# >>> ch7[3]
# {'first-sentence': '_Convolutional Neural Networks_ (CNNs) are all the rage for _c
# omputer vision_ (image processing).\n',
#  'last-sentence': "And there's not a single reference implementation example in th
# e PyTorch or Keras packages themselves!\n"}
# >>> ch7[0]
# {'block_type': 'section',
#  'level': 1,
#  'section_title': 'Finding patterns in words with convolutional neural networks (C
# NNs)'}
# >>> ch7[1]
# {'first-sentence': 'This chapter covers\n', 'last-sentence': ''}
# >>> ch7[2]
# {'first-sentence': 'In this chapter you will unlock the misunderstood superpowers 
# of convolution for Natural Language Processing.\n',
#  'last-sentence': ''}
# >>> ch7[3]
# {'first-sentence': '_Convolutional Neural Networks_ (CNNs) are all the rage for _c
# omputer vision_ (image processing).\n',
#  'last-sentence': "And there's not a single reference implementation example in th
# e PyTorch or Keras packages themselves!\n"}
# ```

puts "Processing '#{chapter_path}'."
struct_path = chapter_path[0..-5] + 'struct.adoc'
puts "Writing summary to:\n    #{struct_path}\n\n"
yaml_path = chapter_path + '.yml'
puts "Writing natural language to:\n    #{yaml_path}\n\n"
struct_output = File.open(struct_path, "w")
yaml_output = File.open(yaml_path, "w")

struct_output.write("\n:toc: left\n:toclevels: 6\n\n")
struct_output.write("++++
  <style>
  .first-sentence {
    text-align: left;
    margin-left: 0%;
    margin-right: auto;
    width: 66%;
    background: Beige;
  }
  .last-sentence {
    text-align: right;
    margin-left: auto;
    margin-right: 0%;
    width: 66%;
    background: AliceBlue;
  }
  </style>
++++\n")
# some color combinations of left-right (first vs. last sentence) that work:
# Beige, AliceBlue
# Lavender, Gainsboro
# LightBlue, Gainsboro

(Asciidoctor.load_file chapter_path).find_by.each do |block|

  if block.context == :paragraph
    # if block.lines.length > 2
      yaml_output.write("-\n")
      struct_output.write("[.first-sentence]\n")
      yaml_output.write("  first-sentence: |+\n")

      struct_output.write(block.lines[0] + "\n\n")
      yaml_output.write("    " + block.lines[0] + "\n")

      struct_output.write("[.last-sentence]\n")
      yaml_output.write("  last-sentence: |+\n")

      last_line_i = block.lines.length - 1
      if last_line_i > 0
        struct_output.write(block.lines[last_line_i] + "\n\n")
        yaml_output.write("    " + block.lines[last_line_i] + "\n")
      end
    # end

  elsif block.context == :section
    struct_output.write("=" * (block.level+1) + " " + block.title + "\n")
    yaml_output.write("-\n")
    yaml_output.write("  block_type: section\n")
    yaml_output.write("  level: #{block.level+1}\n")
    yaml_output.write("  section_title: \"#{block.title}\"\n")

  elsif block.title?
    struct_output.write("."+block.title+"\n\n")
    yaml_output.write("-\n")
    yaml_output.write("  block_type: other\n")
    yaml_output.write("  level: #{block.level}\n")
    yaml_output.write("  section_title: \"#{block.title}\"\n")

  end

  # YAML.dump(block)
  
end

struct_output.close()