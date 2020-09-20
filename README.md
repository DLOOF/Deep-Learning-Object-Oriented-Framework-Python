# Deep Learning Object Oriented Framework with Python

:exclamation: This is our work on a deep learning object oriented framework developed for our Artificial Intelligence emphasis.

:warning: Our written document is in **Spanish**

## Authors
* Alejandro Anzola Ávila [@aanzolaavila](https://github.com/aanzolaavila)
* Juan Andrés Moreno Silva [@JuanMorenoS](https://github.com/JuanMorenoS)

**Director:** MSc. María Irma Díaz Rozo [marirmadir](https://www.linkedin.com/in/marirmadir/)

## Abstract
> Machine Learning is a field of artificial intelligence that has been developing exponentially in the last few years due to the immense amount of data, processing power, and technology innovations. Among the great strengths that this technology offers is the vast amount of problems from different domains and knowledge fields.  It is even possible to reuse knowledge and models obtained after solving a problem from a similar field.
>
> One of the main actors in this revolution is the neural networks technology, which is part of the field of machine learning. This technology lets us solve an infinite amount of problems, from classifying written numbers to source code generation from natural language.
>
> However, this technology has a very steep learning curve. Our solution to this problem is to offer neural network learning resources to help students learn base concepts through experimentation and extension. The developed resources are (i) a learning framework and (ii) a set of problem-solution cases.
>
> The framework lets students build and test neural network models. This framework is designed to aid newcomers interested in the field to understand base concepts, following software engineering best practices.
>
> This framework is built to represent artificial intelligence base concepts, in a way general enough to be extensible to different neural network models. To achieve this, the framework has various extensible components that correspond to different main concepts of neural networks. Among these components are (i) neural network types (deep, recurrent, and convolutional); (ii) activation and loss functions; and (iii) regularization and optimization mechanisms.
>
> In the set of problem-solution cases, there are three classic problems and one specialized problem in the field of hydrology. The classic problems selected are handwritten digits with dense and convolutional networks, and positive-negative text classification with recurrent networks. The hydrology problem was the prediction of flow in the Magdalena river; in this work, we made two models to evaluate them: a multilayer perceptron (MLP for short) and a Long-short Term Memory (LSTM for short) for evaluation.
>
> The cases developed allowed us to test the versatility of our framework in addition to having learning resources. In three of the four cases, the performance of the framework was verified and deemed appropriate. The case that could not be implemented was the one that required recurrent neural networks due to encountered scope problems; however, recurrent neural networks are considered in the framework structure's design.
>
> In the implementation of the cases, it could be seen that the framework works aligned with the philosophy of good software development practices. In particular, in the implementation of the hydrology problem, it was shown that it is easy to extend.
>
> One of the performance metrics that were taken was training times to have a way of comparing them, although it is not part of our objectives. It was not unexpected to find that Keras' performance is considerably superior to the implementation made in our framework.
>
> Finally, one of our framework's key aspects is that it was designed with only supervised models in mind, which only allows it to be extended to supervised models other than neural networks. The addition of unsupervised learning or reinforced learning is future work.

## About our written document
To take a look into our written document, open [TDIA_2020.pdf](https://github.com/DLOOF/Deep-Learning-Object-Oriented-Framework-Python/blob/master/TDIA_2020.pdf) (:warning: in spanish).

One remark about our notation: we used the mathematical notation from the [Deep Learning Book](https://www.deeplearningbook.org/) because we think that it stablishes some common notation among
deep learning researchers (and that was one of their intentions). To access their notation template for LaTeX you can access it [here](https://github.com/goodfeli/dlbook_notation).

## Why did we do this?
We noticed that a lot some of our colleagues and some published work on this research area (be it formal or informal) have some important conceptual errors that
are well known for almost anyone that works as a professional on the field.

### How did we do it?
We wanted to solve this issue by presenting a framework and an implementation that followed these goals:
* Conceptual modules
* High detail
* Clean code

## UML diagram
To compile our UML diagram, you can go into the `model-diagram` folder (make sure to have installed Java) and execute this command:
```bash
$ java -jar plantuml.jar model-diagram.plantuml -tpdf
```
And it will generate the diagram into a PDF called `model-diagram.pdf`.
This was done using [PlantUML](https://plantuml.com/).

## Prerequisites
This project was developed with Python 3. Our complete list of dependencies are included in the [requirements.txt](https://github.com/DLOOF/Deep-Learning-Object-Oriented-Framework-Python/blob/master/requirements.txt) file.

To get a virtual environment working (and not screw up your installed dependencies), use the following command in this cloned repository (MacOS, Linux)
```bash
$ python3 -m venv venv
```
That will create a new folder called `venv`, then you will need to execute this command (we are assuming that you are using Bash or some derivative like Zsh)
```bash
$ source venv/bin/activate
```

After that, get all the dependencies to get it working.
```bash
$ python3 -m pip install -r requirements.txt
```

## Citation
To cite us using BibTeX, you can use this
```
@techreport{AnzolaMoreno2020,
  author = {{Anzola Ávila}, {Alejandro} and {Moreno Silva}, {Juan Andrés}},
  year = {2020},
  title = {Framework para el Aprendizaje de Redes Neuronales Profundas},
  institution = {Escuela Colombiana de Ingeniería Julio Garavito}
}
```

You can also find this work from the official repository [here](https://repositorio.escuelaing.edu.co/handle/001/1236).
