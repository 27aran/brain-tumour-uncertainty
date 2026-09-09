# BrainTumourUncertainty

Ursprünglich 99% Val Accuracy, nach Identifikation und Entfernung von 
~2000 (Near-)Duplikaten über Perceptual Hashing sank sie auf ~94-97%,
was realistischer ist. Ein Restrisiko durch Patient-Level-Leakage (mehrere 
Slices desselben Patienten in unterschiedlichen Splits) besteht, da der Datensatz 
keine Patienten-IDs bereitstellt.