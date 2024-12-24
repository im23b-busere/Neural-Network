# Neural-Network
 Meine ersten versuche an einem neural network (°_°)

Meine Idee mit dem Projekt ist ein einfaches neuronales Netzwerk zur Lösung des klassischen XOR-Problems. Das XOR-Problem ist eine grundlegende Herausforderung im Bereich des maschinellen Lernens, da es nicht linear separierbar ist und somit die Grenzen eines einfachen Perzeptrons aufzeigt. Hier verwenden wir ein mehrschichtiges neuronales Netzwerk (MLP), das aus einer versteckten Schicht besteht, um das Problem zu lösen.

Das Netzwerk wird von Grund auf in Python erstellt, unter Verwendung von `numpy` für numerische Berechnungen.

Bildliche darstellung: 

<img src="https://github.com/user-attachments/assets/a868af24-7632-4bd6-988e-962a30f9143f" alt="Visualization" width="700"/>


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

<img src="https://github.com/user-attachments/assets/c973785e-3ea7-463a-8faf-f276ac02e278" alt="Sigmoid" width="700"/>

Diese Funktion transformiert die kontinuierliche reelle Zahl in einen Bereich von (0, 1).



**Ableitung der Sigmoid Funktion (σ) :**

<img src="https://github.com/user-attachments/assets/e1a894d3-5987-4d55-bae1-15f70f163377" alt="Sigmoid-derivative" width="700"/>

Die Ableitung der Sigmoid-Funktion gibt an, wie empfindlich die Sigmoid-Funktion in einem bestimmten Punkt auf Änderungen ihrer Eingabe reagiert.




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


2. **Rückwärtsdurchlauf (Backpropagation):**
   - Fehlerberechnung:
    `error = Y-output - Zielausgabe`
   - Anpassung der Gewichte und Biases mittels Ableitung der Sigmoid-Funktion.
  

# Fehlerverlauf während des Trainings
Nachfolgend ein Beispiel für die Visualisierung der Fehlerentwicklung mit Lernrate 0.1:

<img src="https://github.com/user-attachments/assets/ddb6801a-5d53-4617-b75a-c8a49d2973c6" alt="Fehlerentwicklung während des Trainings" width="700"/>

In diesem Diagramm wird der Fehler pro Zyklus dargestellt, was die Übereinstimmung des Modells verdeutlicht.
     

## Autor

**[im23b-busere](https://github.com/im23b-busere)**  
Feedback oder Vorschläge? Öffne ein Issue oder erstelle einen Pull-Request!

