run_one() {
  REP="$1"
  time python generate_round0.py --rep $REP
}

run_one "0"
run_one "1"
run_one "2"
run_one "3"
run_one "4"
run_one "5"
run_one "6"
run_one "7"
run_one "8"
run_one "9"
