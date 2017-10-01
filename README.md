# The Author Function: Imitating Grant Allen with Queer Writing Machines

> “I cannot choose but wonder who each is, and why he is here. For one after another I invent a story. It may not be the true story, but at least it amuses me.” — Grant Allen, *The Type-writer Girl.*

This repository contains files for "The Author Function," a project that uses machine learning to imitate the style of Grant Allen (1848-1899), a nineteenth-century author who wrote in a variety of genres and topics, and under various pseudonyms—one of which was cross-gendered (“Olive Pratt Rayner”). In recent years, contemporary scholars have revisited Allen's life and work—in particular, how the clash between Allen's political (e.g. in women's rights) and artistic aspirations and economic pressures greatly influenced not only what he wrote, but how he positioned or presented himself in relation to it (Greenslade and Rodgers; Morton; Warne and Colligan). This conflict makes Allen an interesting case study in performative authorship. As I discuss in ["Context"](context.md/grant-allen-a-case-study), Allen wrote in a cultural moment and milieu that makes him a good case study for mechanical reproduction and its relation to (gendered) authorship, both then and now.

To imitate Allen's writing style, I harnessed the power of artificial neural networks (ANNs), which are computer systems modelled loosely on the structure and behaviour of the human brain. In particular, I used the code module torch-rnn, written by Justin Johnson, to "train" an ANN on Allen's writings. After the training process, I then call on or "sample" the network to generate strings of text based on the texts' unique stylistic features (as learned and represented by the ANN).

"The Author Function" is a modest first step in a larger project of exploring the possibilities of machine learning and imitation in/for cultural or literary research. It draws from work by Kari Krauss and Lisa Samuels and Jerome McGann on subjunctive criticism and systematic alterations of text. In their research, these scholars emphasize the potential of speculation: what could we learn about our object of inquiry (e.g. a work of literature) if we broke it down, remade it, and compared or interpreted it alongside, or even as if it were, the original?

In practice and purpose, this project also shares characteristics with other forms or practices of electronic literature or, more specifically, generative or "computational creative" writing such as Twitter bots ([Goodwin](www.medium.com/artists-and-machine-intelligence/adventures-in-narrated-reality-6516ff395ba3); [Kazemi](http://tinysubversions.com/); [Parrish](www.decontextualize.com); [Sample](www.medium.com/@samplereality/a-protest-bot-is-a-bot-so-specific-you-cant-mistake-it-for-bullshit-90fe10b7fbaa); [Sloan](https://www.robinsloan.com/notes/writing-with-the-machine/)). These creative practices all disarticulate written text or language into component parts before rearranging and recombining them in novel formations. (There are, however, significant differences between machine learning and other generative techniques: see ["Anatomy"](context.md/the-anatomy-of-neural-networks).) Articulated in Victorian-era terms, this project is like conducting a séance with a computer, rather than an Ouija board. Framed as a methodological experiment, "The Author Function" asks whether a kind of computer-assisted forgery can tell us anything of academic, literary, or cultural value—and, if so, what might it say?

[Continue to "Context"](context) or scroll down for descriptions of each section: Context, Composition, Corpus, and Code.

## Table of Contents

### Context
* [The Anatomy of Neural Networks](context.md/#the-anatomy-of-neural-networks)
  * [Enter torch-rnn](context.md/#enter-torch-rnn)
* [Grant Allen: a Case Study](context.md/#grant-allen-a-case-study)
  * [The Anxiety of Authorship](context.md/#the-anxiety-of-authorship)
  * [Queer Writing Machines](context.md/#queer-writing-machines)
  * [Possible Future Directions](context.md/#possible-future-directions)
* [Works Cited](context.md/#works-cited)

### Composition
* [Getting Started](composition.md/#started)
* [Interface and Interpretation](composition.md/#interface-and-interpretation)
* [Optimizing Hyper-parameters](composition.md/#optimizing-hyper-parameters)
  * [A Model's Fit: Training vs. Validation Loss](composition.md/#a-models-fit-training-vs-validation-loss)
  * [Changing the Rate of Descent: Learning Decay](composition.md/#changing-the-rate-of-descent)
  * [Training Speed and Duration: Max Epochs, Batch Size, and Early Stopping](composition.md/#training-speed-and-duration-max-epochs-batch-size-and-early-stopping)
  * [Sequence Length](composition.md/#sequence-length)
  * [Stopping and Starting Training](composition.md/#stopping-and-starting-training)
* [Works Cited](composition.md/#works-cited)

### Corpus

### Code
* [Notes on Running the Scripts, Navigating Directories](code/README.md/#notes-on-running-the-scripts-navigating-directories)
* [The Scripts in More Detail](code/README.md/#the-scripts-in-more-detail)

## Context

This document contextualizes and interprets the project in terms of its cultural, historical, and technological significance or implications. It gives a brief overview of machine/deep learning and discusses how this technology came to be available to non-specialists. It also explains why Grant Allen (1848-1899) makes a good case study, as well as points to possible future directions for research.

<img src="images/Grant-Allensq.jpg" width="30%" /><img src="images/1-layers.png" width="35%" /><img src="images/typist.jpg" width="30%"/>

## Composition

This document provides more detailed and explicit explanations of, and instructions and tips for, machine learning with torch-rnn: a code module developed by Justin Johnson, based on work by Andrej Karpathy.

![The interface for torch-rnn](images/7-interface.png)

## Corpus

This folder contains a collection and description of the 32 .txt files used to train the neural network.

![Image of a corpus file](images/corpus.png)

## Code

This folder contains scripts that I wrote in the programming language Python to preprocess the data (e.g. deleting chapter titles, illustration tags) before feeding it into the neural network.

![Image of a Python script](images/script.png)

## Works Cited

Greenslade, William and Terence Rodgers. “Resituating Grant Allen: Writing, Radicalism, and Modernity.” *Grant Allen: Literature and Cultural Politics at the Fin de Siècle*. Ashgate, 2005, pp. 1-23.

Goodwin, Ross. “Adventures in Narrated Reality: New forms & interfaces for written language, enabled by machine intelligence.” *Artists and Machine Intelligence*, *Medium.com*, www.medium.com/artists-and-machine-intelligence/adventures-in-narrated-reality-6516ff395ba3

Kazemi, Darius. *Tiny Subversions*, n.d. http://tinysubversions.com/

Kraus, Kari. “Conjectural Criticism: Computing Past and Future Texts.” *Digital Humanities Quarterly*, vol. 3, no. 4, 2009, n.p.

Morton, Peter. *“The Busiest Man in England”: Grant Allen and the Writing Trade, 1875-1900*. Palgrave Macmillan, 2005.

Parrish, Allison. *Decontextualize: Allison Parrish: words and projects*, 2016, www.decontextualize.com 

Sample, Mark. “A protest bot is a bot so specific you can’t mistake it for bullshit: A Call for Bots of Conviction.” *Medium.com*, 30 May 2014. www.medium.com/@samplereality/a-protest-bot-is-a-bot-so-specific-you-cant-mistake-it-for-bullshit-90fe10b7fbaa

Samuels, Lisa and Jerome McGann. “Deformance and Interpretation.” *New Literary History*, vol. 30, no. 1, 1999, pp. 25-26.

Sloan, Robin. "Writing with the machine." *Robinsloan.com*, n.d. https://www.robinsloan.com/notes/writing-with-the-machine/

## Version

Will write this later.

## License

Will write this later.

## Acknowledgements

Will write this later.
