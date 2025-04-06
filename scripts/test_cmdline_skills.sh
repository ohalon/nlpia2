#!/usr/bin/env bash
#---- Script to test the commandline versions of the various skills  ------
# Usage: Modify the bash array variable 'skills_to_test' below to test the required skills 

### Details ###
# This will use an input file ${TESTSET_PATH}/test_${skill}.input.txt to give inputs to
# the tested skill and record the skill's output in the file 
# ${TESTSET_PATH}/test${skill}.output.txt. That will be compared with the solution file
# ${TESTSET_PATH}/test${skill}.solution.txt and will pass the test if there is a perfect
# match (ignoring trailing whitespace and EOL differences)

skills_to_test=("qa" "quiz")
TESTSET_PATH="src/qary/data/tests"

#--------------------- bash Function that does the actual test ---------------------
# argument is the skill to be tested
function run_cmdline_test() {
    skill=$1
    input_filepath="${TESTSET_PATH}/test_${skill}.input.txt"
    output_filepath="${TESTSET_PATH}/test_${skill}.output.txt"
    solution_filepath="${TESTSET_PATH}/test_${skill}.solution.txt"
    if [ ! -f $input_filepath ]; then
        echo "Missing ${input_filepath} for ${skill} test; exiting..."
        return 1
    fi
    if [ ! -f $solution_filepath ]; then
        echo "Missing ${solution_filepath} for ${skill} test; exiting..."
        return 1
    fi
    qary -s $skill < "${input_filepath}" > "${output_filepath}"
    diff_output=$(diff --ignore-trailing-space $output_filepath $solution_filepath)
    if [ ! -z "${diff_output}" ]; then
        echo "Commndline skill ${skill} failed..."
        echo "Following mismatch between expected and actual output was found..."
        echo $diff_output
        return 1
    else
        echo "Commndline skill ${skill} passed"
        return 0
    fi
}

#--------------------- Run cmdline test on specified skills  ---------------------
exit_codes=()
echo "Following skills will be tested on the cmdline: ${skills_to_test[@]}"
for skill in ${skills_to_test[@]}; do
    echo "Running commandline test for skill ${skill}..."
    run_cmdline_test $skill
    exit_codes+=($?)
    echo $'\n'
done
#------------ Compute OR of all exit codes and report failed tests ----------------
final_exit_code=0
for i in ${!exit_codes[@]}; do
    if [ ${exit_codes[$i]} -ne 0 ]; then
        final_exit_code=1
    fi
done
exit $final_exit_code

