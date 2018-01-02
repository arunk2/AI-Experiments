```
<?php

# Compute signatures for two images
$cvec1 = puzzle_fill_cvec_from_file('/home/dev/test/1.jpg');
$cvec2 = puzzle_fill_cvec_from_file('/home/dev/test/2.jpg');

# Compute the distance between both signatures
$d = puzzle_vector_normalized_distance($cvec1, $cvec2);

echo "Score = ".$d." : thres : ". PUZZLE_CVEC_SIMILARITY_LOWER_THRESHOLD."\n";
# Are pictures similar?
if ($d < PUZZLE_CVEC_SIMILARITY_LOWER_THRESHOLD) {
  echo "Pictures are looking similar\n";
} else {
  echo "Pictures are different, distance=$d\n";
}

# Compress the signatures for database storage
$compress_cvec1 = puzzle_compress_cvec($cvec1);
$compress_cvec2 = puzzle_compress_cvec($cvec2);

?>
```
