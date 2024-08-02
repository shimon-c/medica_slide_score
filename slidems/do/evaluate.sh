#!/bin/bash

source ./do/common.sh


DellIfExists ${WS}/build
ActivateVENV "$venvName"


#$py -m slidems.slide.evaluate_focus -i /home/dudi/privet/med/openslide/output_images/0305 -o /home/dudi/privet/med/biopsyfocus/output/0305
#$py -m slidems.evaluate_slide -i /home/dudi/privet/med/raw_data/good -o /home/dudi/privet/med/biopsyfocus/output/good
#$py -m slidems.evaluate_slide -i /home/dudi/privet/med/raw_data/bad/ANON0LPGUI17B_1_1.ndpi -o /home/dudi/privet/med/biopsyfocus/output/OOF
#[ $? -ne 0 ] && ExitError

#$py -m slidems.evaluate_slide -i /home/dudi/privet/med/raw_data/bad -o /home/dudi/privet/med/biopsyfocus/output/bad
#[ $? -ne 0 ] && ExitError

# $py -m slidems.evaluate_slide -i /home/dudi/privet/med/raw_data/dubleScan1/ANONKMHBUI1RK_1_1.ndpi -o /home/dudi/privet/med/biopsyfocus/output/goodd
# [ $? -ne 0 ] && ExitError
# $py -m slidems.evaluate_slide -i /home/dudi/privet/med/raw_data/dubleScan1/ANONKMHBUI1RK_2_1.ndpi -o /home/dudi/privet/med/biopsyfocus/output/goodd
# [ $? -ne 0 ] && ExitError
# $py -m slidems.evaluate_slide -i /home/dudi/privet/med/raw_data/dubleScan2/ANONS0IBUI175_1_1.ndpi -o /home/dudi/privet/med/biopsyfocus/output/goodd
# [ $? -ne 0 ] && ExitError
# $py -m slidems.evaluate_slide -i /home/dudi/privet/med/raw_data/dubleScan3/01-01_01_HE_01-01-01_0/ANONF2IBUI1F2_1_1.ndpi -o /home/dudi/privet/med/biopsyfocus/output/goodd
# [ $? -ne 0 ] && ExitError
# $py -m slidems.evaluate_slide -i /home/dudi/privet/med/raw_data/dubleScan3/01-01_01_HE_01-01-01_1/ANONF2IBUI1F2_2_1.ndpi -o /home/dudi/privet/med/biopsyfocus/output/goodd
# [ $? -ne 0 ] && ExitError
$pyt -m slidems.evaluate_slide -i /home/dudi/dev/pathology/raw_data/bad/ANON0LPGUI17B_1_1.ndpi -o /home/dudi/dev/pathology/slidems/output/goodd
[ $? -ne 0 ] && ExitError


ExitOk