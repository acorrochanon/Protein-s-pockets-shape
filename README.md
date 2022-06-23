# Learning the shapes of protein pockets

<p align="justify">
The comparison of protein pockets plays an important role in drug discovery. Through the identification of binding sites with similar structures, we can assist in finding hits and characterizing the function of proteins. Traditionally, the geometry of cavities has been described with scalar representations that can be considered not accurate, as they do not fully account for the shape as a whole. In this work, we propose a method that creates geometrical descriptions of pocket shape based on Euclidean neural networks, allowing us to encode their physical features. By doing this, we can compare the cavities through the computation of the Euclidean distance between their respective embeddings. In order to ensure these embeddings contain relevant geometrical information, our model has been trained on a supervised classification task to predict whether given pockets are druggable. 
</p>
 
<p align="center">
  <img src="https://user-images.githubusercontent.com/56264560/175291037-15fd52df-5225-46bf-b7a0-f46a5c7c8b7b.png" width="600" height="250"/>
</p>
 
<p align="justify">
To achieve this, a new dataset was built from the existing sc-PDB database that served as a reference to set the druggable cavities. Then, the protein cavity detection algorithm Fpocket was applied to generate decoys. The supervised model is evaluated by predicting druggability on held-out data, while the utility of the learned embeddings is assessed by comparing how a pocket changes during a dynamic simulation. The findings obtained are encouraging and suggest a possible paradigm shift in the way binding sites comparison can be approached.
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/56264560/175293821-079da307-67ee-4b32-859e-21fc8fe641eb.png" width="350" height="200" alt>
  <img src="https://user-images.githubusercontent.com/56264560/175293665-2d25206c-07f5-4c70-8087-5d022d2d0a1a.png" width="350" height="200" alt>
</p>

<p align="justify">
<b>This project was developed as a master's thesis by Alejandro Corrochano and Yossra Gharbi. In collaboration with Jon Paul Janet (AstraZeneca).</b>
</p>
