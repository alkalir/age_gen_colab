dos2unix_conversion() {
  #dos2unix conversion
  for file in $(find $PWD -name "*.sh"); do
      sed -i 's/\r//g' ${file}
      echo  ${file}
  done
}

1_AgeGen_quant() {
  echo " "
  echo "##########################################################################"
  echo "QUANTIZE Age"
  echo "##########################################################################"
  python quantizationAgeGen_tf2.py -n Age 2>&1 | tee target_zcu102/rpt/AgeGenRec/1_quantizationAge_tf2_3.log
  echo " "
  echo "##########################################################################"
  echo "QUANTIZE Gen"
  echo "##########################################################################"
  python quantizationAgeGen_tf2.py -n Gen 2>&1 | tee target_zcu102/rpt/AgeGenRec/1_quantizationGen_tf2.log

  echo " "
  echo "##########################################################################"
  echo "QUANTIZATION COMPLETED"
  echo "##########################################################################"
  echo " "
}


2_AgeGen_evaluate_quantized_model() {
  echo " "
  echo "##########################################################################"
  echo "EVALUATE QUANTIZED GRAPH of Age on Morph"
  echo "##########################################################################"
  python eval_quantization_AgeGen.py -n Age  2>&1 | tee target_zcu102/rpt/AgeGenRec/2_evaluate_quantized_model_Age.log

  echo " "
  echo "##########################################################################"
  echo "EVALUATE QUANTIZED GRAPH of Gen on Morph"
  echo "##########################################################################"
  python eval_quantization_AgeGen.py -n Gen  2>&1 | tee target_zcu102/rpt/AgeGenRec/2_evaluate_quantized_model_Gen.log

  echo " "
  echo "##########################################################################"
  echo "EVALUATE QUANTIZED GRAPH COMPLETED on Morph"
  echo "##########################################################################"
  echo " "

}

3_AgeGen_dump(){
  echo " "
  echo "##########################################################################"
  echo "DUMP QUANTIZED MODEL of Age on Morph"
  echo "##########################################################################"
  python dump_quantized_model.py -n Age  2>&1 | tee target_zcu102/rpt/AgeGenRec/3_dump_Age.log

  echo " "
  echo "##########################################################################"
  echo "DUMP QUANTIZED MODEL of Gen on Morph"
  echo "##########################################################################"
  python dump_quantized_model.py -n Gen  2>&1 | tee target_zcu102/rpt/AgeGenRec/3_dump_Gen.log

}

4_AgeGen_vai_compile_zcu102() {
echo " "
echo "##########################################################################"
echo "COMPILE WITH Vitis AI on ZCU102: Age on Morph"
echo "##########################################################################"
vai_c_tensorflow2 \
       --model ./build/quantized_results/AgeGen/Age/quantized_model.h5 \
       --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json \
       --output_dir=./build/compile/AgeGen/Age \
       --net_name=Age \
       --options    "{'mode':'normal'}" \
       2>&1 | tee target_zcu102/rpt/AgeGenRec/4_vai_compile_Age.log

mv  ./build/compile/AgeGen/Age/*.xmodel ./target_zcu102/AgeGen/Age

echo " "
echo "##########################################################################"
echo "COMPILE WITH Vitis AI on ZCU102: Gen on Morph"
echo "##########################################################################"
vai_c_tensorflow2 \
       --model ./build/quantized_results/AgeGen/Gen/quantized_model.h5 \
       --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json \
       --output_dir=./build/compile/AgeGen/Gen \
       --net_name=Gen \
       --options    "{'mode':'normal'}" \
       2>&1 | tee target_zcu102/rpt/AgeGenRec/4_vai_compile_Gen.log

mv  ./build/compile/AgeGen/Gen/*.xmodel ./target_zcu102/AgeGen/Gen

echo " "
echo "##########################################################################"
echo "COMPILATION COMPLETED on Morph on ZCU102"
echo "##########################################################################"
echo " "
}


main() {

  # quantize the CNN from 32-bit floating-point to 8-bit fixed-point
  1_AgeGen_quant

  # make predictions with quantized frozen graph
  2_AgeGen_evaluate_quantized_model

  # compare the simulation results on the CPU/GPU and the output values on the DPU.
  3_AgeGen_dump

  # compile xmodel file for ZCU102 target board
  4_AgeGen_vai_compile_zcu102
 
  #python code/generation_test_image.py
  ## copy test images into target board
  #cd build/dataset/morph
  #cp -r ./test ./morph_test
  #tar -cvf morph_test.tar ./morph_test &> /dev/null
  #rm -rf morph_test
  #cd ../../../
  #cp ./build/dataset/morph/morph_test.tar ./target_zcu102/
  #tar -cvf target_zcu102.tar ./target_zcu102 &> /dev/null


}

main
