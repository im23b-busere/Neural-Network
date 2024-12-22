# Neural-Network
 Meine ersten versuche an einem neural network (°_°)

Meine Idee mit dem Projekt ist ein einfaches neuronales Netzwerk zur Lösung des klassischen XOR-Problems. Das XOR-Problem ist eine grundlegende Herausforderung im Bereich des maschinellen Lernens, da es nicht linear separierbar ist und somit die Grenzen eines einfachen Perzeptrons aufzeigt. Hier verwenden wir ein mehrschichtiges neuronales Netzwerk (MLP), das aus einer versteckten Schicht besteht, um das Problem zu lösen.

Das Netzwerk wird von Grund auf in Python erstellt, unter Verwendung von `numpy` für numerische Berechnungen.

Bildliche darstellung: 

<img src="https://github.com/user-attachments/assets/a868af24-7632-4bd6-988e-962a30f9143f" alt="AI-Voice-Assistant-Screenshot" width="700"/>


Das XOR-Problem besteht darin, die Ausgabe einer exklusiven Oder-Operation (XOR) zu modellieren:

| Input x1 | Input x2 | Zielausgabe (y)  |
|------------|------------|------------------|
| 0          | 0          | 0                |
| 0          | 1          | 1                |
| 1          | 0          | 1                |
| 1          | 1          | 0                |





---

## Mathematischer Hintergrund

### 1. Aufbau des Netzwerks
- **Eingabeschicht**: 2 Neuronen (entspricht den Eingaben x1 und x2).
- **Versteckte Schicht**: 2 Neuronen mit Sigmoid-Aktivierung.
- **Ausgabeschicht**: 1 Neuron mit Sigmoid-Aktivierung.

**Sigmoid Funtktion (σ) :**

<img src="https://github.com/user-attachments/assets/c973785e-3ea7-463a-8faf-f276ac02e278" alt="AI-Voice-Assistant-Screenshot" width="700"/>



Diese Funktion transformiert die kontinuierliche reelle Zahl in einen Bereich von (0, 1)

Das Netzwerk berechnet die Ausgabe wie folgt:

1. **Vorwärtsdurchlauf (Forward Pass):**
   - Berechnung der gewichteten Summe in der versteckten Schicht:
     ` = (X * W) +B`
     
   - Aktivierung der versteckten Schicht mit Sigmoid Funktion:
      ` H = σ ((X * W) + B)`
     
   - Berechnung der gewichteten Summe in der Ausgabeschicht:
     `Y-input = (H * W-output) + B-output` 
   - Aktivierung der Ausgabeschicht:
     `Y-output = σ (Y-input)` 


