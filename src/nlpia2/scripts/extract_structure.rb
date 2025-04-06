#!/usr/bin/env ruby
# ARGV.each do|a|
#   puts "Argument: #{a}"
# end
chapter = 'manuscript/adoc/Chapter 07 -- Getting Words in Order with Convolutional Neural Networks (CNNs).adoc'    # replace 

if ARGV.length == 1
  # chapter, *unused_argv = ARGV
  chapter = ARGV.at(0)
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
# How I use it:
# set the chapter name (future: pass as parameter), run the script, and with the Chrome extension 
# "Asciidoctor.js Live Preview", browse the struct_flnm to review the structure.

require 'asciidoctor'

puts "Processing '#{chapter}'."
struct_flnm = chapter[0..-5] + 'struct.adoc'
puts "Writing summary to '#{struct_flnm}'..."
output = File.open(struct_flnm, "w")

output.write("\n:toc: left\n:toclevels: 6\n\n")
output.write("++++
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

(Asciidoctor.load_file chapter).find_by.each do |block|
  if block.context == :paragraph
    output.write("[.first-sentence]\n")
    output.write(block.lines[0] + "\n\n")
    output.write("[.last-sentence]\n")
    output.write(block.lines[(block.lines.length) -1] + "\n\n")
  elsif block.context == :section
    output.write("=" * (block.level+1) + " " + block.title + "\n")
  elsif block.title?
    output.write("."+block.title+"\n\n")
  end
end

output.close()