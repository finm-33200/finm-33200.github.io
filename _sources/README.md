Course Syllabus: FINM 32900, Winter 2025
========================================

**FINM 33200, Generative and Agentic AI for Finance**

##  Summary

**Course Description** "Generative and Agentic AI for Finance" is a hands-on course covering the practical applications of generative AI and agentic technologies in quantitative finance. This course provides hands-on experience with generative AI and agentic technologies and their practical applications in quantitative finance. Students will start by mastering modern AI development tools like Cursor editor and GitHub Copilot, then progress through understanding AI tool use, building autonomous agents, and implementing sophisticated systems like RAG. The course emphasizes real-world implementation using cutting-edge AI frameworks, covering everything from LLM fundamentals and the Model Context Protocol to building custom AI applications and deploying them in production environments. By the end of the course, students will be proficient in creating production-ready AI applications that can transform financial workflows and decision-making processes through intelligent automation and enhanced data analysis.

- **Class:** TBD, in-person at the Stevanovich Center building,
  Room #112. (5727 S. University Ave.)
- **Lecturers:** 
  - Mark Hendricks, mhendricks@uchicago.edu
  - Jeremy Bejarano, jbejarano@uchicago.edu
- **Instructor Office Hours:** TBD, in the FinMath library (first floor of the Stevanovich Center, 5727 S. University Ave.)
- **Teaching Assistants:**
  - TBD
  - Note: Please include both TAs on all emails. However, students are strongly
    encouraged to post questions on the discussion page of the class GitHub
    repository here.

- **TA Session/TA Office Hours:** TBD, on Zoom. See the Zoom link on the Canvas calendar.

- **Website:** Canvas will be used for grades and for publishing Zoom links
  only. Homework and notes will be posted on the course GitHub repo:
  https://github.com/finm-32900/finm-32900-data-science. Questions and other
  class-related discussions should be posted here as well.
- **Textbook:** The text for the course will be published incrementally here:
  https://finm-32900.github.io/

**NOTE:** Due to the holiday on January 20, a makeup class will be scheduled. This makeup class will be held on Friday, January 24 during the TA office hours. I will use the same Zoom link as the TA office hours, which you can find on the Canvas calendar.


### Assignments

- Assignments must be submitted via GitHub before 6 pm on Fridays. Each
  assignment will be distributed on a Monday, and will be due the on the Friday
  of the following week (11 days later).
- Assignments are automatically graded via the autograder on GitHub Classroom
  and solutions will be released shortly after. This means that the due date is
  strict. Late assignments will not be accepted.
- Each student is to individually submit their assignment (unless otherwise
  specified). Students are encouraged to work in groups, but students are not
  allowed to copy each other's code. Each student must write their own solutions
  individually.
- After assignments are graded, solutions will be posted in separate GitHub
  repos, found here: https://github.com/finm-32900

### Final Project

In lieu of a final exam, students will be organized into groups of 2 (pairs) and
will each complete a course project. Each group will present their completed
project to the instructor at the end of the course. These presentations will be
scheduled individually. 

## Assessment

Grades will be based on coding assignments (40%), a final group project (55%),
and participation (5%). 

- Assignments will be submitted individually and will be graded using GitHub’s
  automated testing tools. 
- The final project will be completed in groups. Students will choose the
  project from among a few options provided at the beginning of the quarter. The
  project will be graded not only on how well it accomplishes the assigned data
  cleaning and analysis task, but will be primarily graded on whether (1) the
  steps to reproduce it are fully automated and well documented, (2) the code is
  written in a clean and reusable fashion, and (3) the results are presented
  clearly and presented in a way that convinces the reader that the results are
  correct. A more specific rubric will be provided in class.
- The participation grade will depend on the positive impacts that a student has
  on the class. These include participating in in-class discussions and/or
  answering questions on the class GitHub page (or on Canvas). Students are in
  no way penalized for giving wrong answers in these in-class discussions nor is
  there any penalty for asking for help—asking for help is often the best way to
  learn!


## Schedule

The schedule will follow the ordering of the chapters listed in the GitHub book
found here: https://finm-32900.github.io/. Each week is it's own chapter and the
agenda is listed in the first sub-section of the chapter.

### HW Due Dates

- [HW 0: Ungraded. Due ASAP](HW0.md)


## Quick Start

To quickest way to run code in this repo is to use the following steps. First, you must have the `conda`  
package manager installed (e.g., via Anaconda). However, I recommend using `mamba`, via [miniforge]
(https://github.com/conda-forge/miniforge) as it is faster and more lightweight than `conda`. Second, you 
must have TexLive (or another LaTeX distribution) installed on your computer and available in your path.
You can do this by downloading and 
installing it from here ([windows](https://tug.org/texlive/windows.html#install) 
and [mac](https://tug.org/mactex/mactex-download.html) installers).
Having done these things, open a terminal and navigate to the root directory of the project and create a 
conda environment using the following command:
```
conda create -n finm python=3.12
conda activate finm
```
and then install the dependencies with pip
```
pip install -r requirements.txt
```
Finally, you can then run 
```
doit
```
And that's it! The landing page of the textbook website will be available at `./_build/html/index.html`.

