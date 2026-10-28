# Confusion matrices (test split; rows = gold, cols = predicted)

## dummy_always_changed

2x2 label (['preserved', 'changed'])

```
[[  0 140]
 [  0 140]]
```

7x7 dimension (['none', 'negation', 'quantity', 'deadline', 'intent', 'entity', 'multi'])

```
[[140   0   0   0   0   0   0]
 [ 28   0   0   0   0   0   0]
 [ 28   0   0   0   0   0   0]
 [ 28   0   0   0   0   0   0]
 [ 28   0   0   0   0   0   0]
 [ 28   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0]]
```

## dummy_always_preserved

2x2 label (['preserved', 'changed'])

```
[[140   0]
 [140   0]]
```

7x7 dimension (['none', 'negation', 'quantity', 'deadline', 'intent', 'entity', 'multi'])

```
[[140   0   0   0   0   0   0]
 [ 28   0   0   0   0   0   0]
 [ 28   0   0   0   0   0   0]
 [ 28   0   0   0   0   0   0]
 [ 28   0   0   0   0   0   0]
 [ 28   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0]]
```

## translation_pipeline

2x2 label (['preserved', 'changed'])

```
[[126  14]
 [  0 140]]
```

7x7 dimension (['none', 'negation', 'quantity', 'deadline', 'intent', 'entity', 'multi'])

```
[[140   0   0   0   0   0   0]
 [ 28   0   0   0   0   0   0]
 [ 28   0   0   0   0   0   0]
 [ 28   0   0   0   0   0   0]
 [ 28   0   0   0   0   0   0]
 [ 28   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0]]
```

