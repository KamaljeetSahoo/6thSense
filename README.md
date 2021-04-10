# 6thSense

Find all the codes to run and build all the features of 6th Sense technologies.






<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email, project_title, project_description
-->




[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/KamaljeetSahoo/6thSense">
    <img src="logo.jpeg" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">6th Sense</h3>

</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#requirements">Requirement</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

**Our Product The Sixth-Sense will serve as an interface cum assistant for the specially-abled individuals primarily suffering from visual impairment, hearing impairment and muteness.6th sense aims to bridge the gap between such individuals and the real world along with its challenges upto a level that hasn’t been achieved yet.**



### Built With

* []()
* []()
* []()



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Requirement

Run the below code in the terminal of your environment.

  ```sh
  pip install -r requirements.txt 
  ```





<!-- USAGE EXAMPLES -->
## Instructions 

Follow the Below Instructiins to use the codes as features of this repo.
 
### Hand Sign Langauge Detection
 
The folder Hand_symbol_Detection contains the code for the same.

**Epoch.pt** contains the trained model weights to run the model.

Run **Main.py** to get the hand symbol detected output from a given hand image.


### Pose Estimation
 
The folder Pose_Estimation_Mobilenet contains the code for the same.

**utils** folder contains the training utils to run and detect body pose.

**weights** folder ontains the trained model weights to run the model.

**Model.py** contains the code for model architrecture

Run **Video.py** to get the pose estimation and a roi around the hand which coould be used for hand symbol recognition.

### GUI
 ```sh
  cd Dashboard
  ```
### DASHBOARD
 ```sh
  python main.py
  ```
## Text and speech transformation

**text2speech_and_speech2text.ipnyb** contains the code for speech and text conversion.


### OCR

**Model.py** contains the code for the ocr to get characters from an image 





<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* []()
* []()
* []()





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
