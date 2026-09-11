# BrainTumourUncertainty

Ursprünglich 99% Val Accuracy, nach Identifikation und Entfernung von 
~2000 (Near-)Duplikaten über Perceptual Hashing sank sie auf ~94-97%,
was realistischer ist. Ein Restrisiko durch Patient-Level-Leakage (mehrere 
Slices desselben Patienten in unterschiedlichen Splits) besteht, da der Datensatz 
keine Patienten-IDs bereitstellt.

Das trainierte Modell zeigte bereits eine gute Kalibrierung (ECE=0.0168). Temperature Scaling (T=1.05) erzielte nur 
eine marginale Verbesserung (ECE=0.0159), was nahelegt, dass Standard-Cross-Entropy-Training hier nicht zu starker 
Overconfidence geführt hat – möglicherweise begünstigt durch Transfer Learning mit einem bereits gut vortrainierten Backbone."