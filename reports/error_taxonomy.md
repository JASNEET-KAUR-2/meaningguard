# Error taxonomy

_Auto-generated from predictions/muril_base_test.jsonl: the 20 most-confident wrong predictions out of 112 total errors. Failure modes are grouped mechanically by (error type, gold dimension, edit-operation family); the analysis paragraph of each group must be written by hand after reading the examples._

## Failure mode 1: wrong_dimension | gold dimension: intent | edit op family: int_imperative  (n=7 of top-20)

- `S0204_3` (conf 0.61)
  - S1: प्लीज़ फॉर्म भरना है?
  - S2: प्लीज़ फॉर्म भरो
  - gold: changed/intent | edit: int_imperative_to_question | model: changed/negation
- `S0163_3` (conf 0.61)
  - S1: प्लीज़ दस्तावेज़ भेजना है?
  - S2: प्लीज़ दस्तावेज़ भेजो
  - gold: changed/intent | edit: int_imperative_to_question | model: changed/negation

**Analysis:** TBD (write 1 paragraph after reading the examples above).

## Failure mode 2: wrong_dimension | gold dimension: quantity | edit op family: qty_word_swap_teen  (n=3 of top-20)

- `S0066_4` (conf 0.61)
  - S1: sau seat bhejo
  - S2: teen seat bhejo
  - gold: changed/quantity | edit: qty_word_swap_teen_to_sau | model: changed/negation
- `S0231_3` (conf 0.61)
  - S1: तीन फाइल रखो
  - S2: सौ फाइल रखो
  - gold: changed/quantity | edit: qty_word_swap_teen_to_sau | model: changed/negation

**Analysis:** TBD (write 1 paragraph after reading the examples above).

## Failure mode 3: wrong_dimension | gold dimension: quantity | edit op family: qty_word_swap_ek  (n=2 of top-20)

- `S0164_4` (conf 0.61)
  - S1: एक फाइल देखो
  - S2: तीन फाइल देखो
  - gold: changed/quantity | edit: qty_word_swap_ek_to_teen | model: changed/negation
- `S0164_3` (conf 0.61)
  - S1: दस फाइल देखो
  - S2: एक फाइल देखो
  - gold: changed/quantity | edit: qty_word_swap_ek_to_das | model: changed/negation

**Analysis:** TBD (write 1 paragraph after reading the examples above).

## Failure mode 4: wrong_dimension | gold dimension: quantity | edit op family: qty_word_swap_das  (n=2 of top-20)

- `S0051_3` (conf 0.61)
  - S1: das package bhejo
  - S2: teen package bhejo
  - gold: changed/quantity | edit: qty_word_swap_das_to_teen | model: changed/negation
- `S0051_4` (conf 0.61)
  - S1: ek package bhejo
  - S2: das package bhejo
  - gold: changed/quantity | edit: qty_word_swap_das_to_ek | model: changed/negation

**Analysis:** TBD (write 1 paragraph after reading the examples above).

## Failure mode 5: wrong_dimension | gold dimension: quantity | edit op family: qty_times10  (n=1 of top-20)

- `S0262_4` (conf 0.61)
  - S1: 1000 रुपये भेजो
  - S2: 10000 रुपये भेजो
  - gold: changed/quantity | edit: qty_times10 | model: changed/negation

**Analysis:** TBD (write 1 paragraph after reading the examples above).

## Failure mode 6: wrong_dimension | gold dimension: entity | edit op family: ent_swap_axis  (n=1 of top-20)

- `S0276_3` (conf 0.61)
  - S1: Axis account me loan jama karo
  - S2: HDFC account me loan jama karo
  - gold: changed/entity | edit: ent_swap_axis_to_hdfc | model: changed/negation

**Analysis:** TBD (write 1 paragraph after reading the examples above).

## Failure mode 7: wrong_dimension | gold dimension: quantity | edit op family: qty_div10  (n=1 of top-20)

- `S0262_3` (conf 0.61)
  - S1: 100 रुपये भेजो
  - S2: 1000 रुपये भेजो
  - gold: changed/quantity | edit: qty_div10 | model: changed/negation

**Analysis:** TBD (write 1 paragraph after reading the examples above).

## Failure mode 8: wrong_dimension | gold dimension: entity | edit op family: ent_swap_icici  (n=1 of top-20)

- `S0003_3` (conf 0.61)
  - S1: एचडीएफसी अकाउंट में बिल जमा करो
  - S2: आईसीआईसीआई अकाउंट में बिल जमा करो
  - gold: changed/entity | edit: ent_swap_icici_to_hdfc | model: changed/negation

**Analysis:** TBD (write 1 paragraph after reading the examples above).

## Failure mode 9: wrong_dimension | gold dimension: entity | edit op family: ent_swap_priya  (n=1 of top-20)

- `S0192_4` (conf 0.61)
  - S1: Please send the report to Rohan
  - S2: Please send the report to Priya
  - gold: changed/entity | edit: ent_swap_priya_to_rohan | model: changed/negation

**Analysis:** TBD (write 1 paragraph after reading the examples above).

## Failure mode 10: wrong_dimension | gold dimension: deadline | edit op family: dl_drop_tak  (n=1 of top-20)

- `S0283_4` (conf 0.61)
  - S1: बिल आज तक भेजो
  - S2: बिल आज भेजो
  - gold: changed/deadline | edit: dl_drop_tak | model: changed/negation

**Analysis:** TBD (write 1 paragraph after reading the examples above).

