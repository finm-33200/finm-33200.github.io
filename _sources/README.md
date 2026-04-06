Course Syllabus: FINM 33200, Spring 2026
========================================

**FINM 33200, Generative and Agentic AI for Finance**

##  Summary

**Course Description** "Generative and Agentic AI for Finance" is a hands-on course covering the practical applications of generative AI and agentic technologies in quantitative finance. This course provides hands-on experience with generative AI and agentic technologies and their practical applications in quantitative finance. Students will start by mastering modern AI development tools like Cursor editor and GitHub Copilot, then progress through understanding AI tool use, building autonomous agents, and implementing sophisticated systems like RAG. The course emphasizes real-world implementation using cutting-edge AI frameworks, covering everything from LLM fundamentals and the Model Context Protocol to building custom AI applications and deploying them in production environments. By the end of the course, students will be proficient in creating production-ready AI applications that can transform financial workflows and decision-making processes through intelligent automation and enhanced data analysis.

- **Class:** 9:30-12:30, in-person at the Stevanovich Center building,
  Room #112. (5727 S. University Ave.)
- **Lecturers:** 
  - Jeremy Bejarano, jbejarano@uchicago.edu
  - Mark Hendricks, hendricks@uchicago.edu
- **Instructor Office Hours:** 1:30-2:30pm, in the FinMath library (first floor of the Stevanovich Center, 5727 S. University Ave.)

- **Online Office Hours:** TBD, on Zoom. See the Zoom link on the Canvas calendar.

- **Website:** Canvas will be used for grades and for publishing Zoom links
  only. Homework and notes will be posted on the course GitHub repo:
  https://github.com/finm-33200/finm-33200-data-science. Questions and other
  class-related discussions should be posted here as well.
- **Textbook:** The text for the course will be published incrementally here:
  https://finm-33200.github.io/



### Prerequisites

Prior completion of FINM 32900 (Full-Stack Quantitative Finance) is recommended but not required. Most students will not have taken that course. Regardless, students are expected to be comfortable with the following tools and concepts on their own:

- **Git and GitHub:** Version control basics (cloning, committing, pushing, pulling, branching), creating and managing repositories, and submitting assignments via GitHub Classroom.
- **GitKraken (or similar Git GUI):** Using a graphical Git client to visualize branches and resolve merge conflicts.
- **The command line:** Navigating the filesystem, running scripts, installing packages, and working in a terminal (bash/zsh).
- **Environment variables and `.env` files:** Setting and managing environment variables, using `.env` files to store API keys securely, and understanding why secrets should not be committed to version control.
- **Task runners:** Using tools like [PyDoit](https://pydoit.org/) to automate build and workflow tasks.
- **Python:** Advanced Python proficiency, including working with virtual environments, `pip`, and common data science libraries.

These topics will not be covered in class. Students who are unfamiliar with any of these tools should review them independently before the course begins.

### Assignments

- Assignments for the first half of the course must be submitted via GitHub. 
- Assignments are automatically graded via the autograder on GitHub Classroom
  and solutions will be released shortly after. This means that the due date is
  strict. Late assignments will not be accepted.
- Each student is to individually submit their assignment (unless otherwise
  specified). Students are encouraged to work in groups, but students are not
  allowed to copy each other's code. Each student must write their own solutions
  individually.
- After assignments are graded, solutions will be posted in separate GitHub
  repos, found here: https://github.com/finm-33200

### Exams

There will be two exams: a midterm and a final. Both are closed-book, multiple-choice, timed, proctored exams with no internet access.

- **Midterm Exam:** The midterm will take place at the start of class during Week 5 (April 20), from 9:30am to 10:30am sharp. The guest seminar will follow from 10:45am to 11:45am.
- **Final Exam:** The final exam will be held during finals week (Week 10).

### Guest Seminar Speakers

During Weeks 5, 6, 7, and 8, class will feature guest seminar speakers. Attendance at these seminars is required and will be taken. Seminar attendance is graded on a pass/fail basis (present or absent). Students are permitted to miss at most one seminar without penalty; any additional absences will affect the seminar attendance grade.

Note that seminar attendance is separate from the participation grade.

### Final Project

Students will form groups of 4 and each complete a course project.
Each group will present their completed project to other student groups and to the instructor at the end of the course. These presentations will be
scheduled individually.

## Assessment

| Component           | Weight |
|---------------------|--------|
| Homework            | 15%    |
| Midterm Exam        | 20%    |
| Final Exam          | 30%    |
| Final Project       | 25%    |
| Seminar Attendance  | 5%     |
| Participation       | 5%     |

- **Homework:** Assignments will be submitted individually and will be graded using GitHub’s
  automated testing tools.
- **Midterm & Final Exams:** Closed-book, multiple-choice, proctored exams. See the Exams section above for scheduling details.
- **Final Project:** The final project will be completed in groups. Students will either choose the
  project from among a few options provided at the beginning of the quarter or they will propose their own project.
- **Seminar Attendance:** Graded based on attendance at the guest seminar sessions during Weeks 5–8. Students may miss one seminar without penalty.
- **Participation:** The participation grade will depend on the positive impacts that a student has
  on the class. These include participating in in-class discussions and/or
  answering questions on the class GitHub page (or on Canvas). Students are in
  no way penalized for giving wrong answers in these in-class discussions nor is
  there any penalty for asking for help—asking for help is often the best way to
  learn!


## Schedule

The schedule will follow the ordering of the chapters listed in the GitHub book
found here: https://finm-33200.github.io/. Each week is its own chapter and the
agenda is listed in the first sub-section of the chapter.

| Week | Date   | Topic                                                                 | Notes              |
|------|--------|-----------------------------------------------------------------------|---------------------|
| 1    | Mar 23 | LLMs in Finance, AI Copilots, APIs & Structured Outputs              | HW0 due             |
| 2    | Mar 30 | Text Representation, Tokenization & Embeddings                       | HW1 released        |
| 3    | Apr 6  | LLM Fundamentals: Chronologically Consistent LLMs, Training from Scratch (nanoGPT, nanochat) | HW2 released |
| 4    | Apr 13 | Agentic Workflows, Tool Use & MCP                                    |                     |
| 5    | Apr 20 | **Midterm Exam** (9:30–10:30am) + Guest: Matt Stockton — *RAG and Agentic Retrieval* (10:45–11:45am) | Midterm |
| 6    | Apr 27 | Guest: Brian Lewis — *Where Most AI Tools Fail in Large Enterprises and Why* (11:00am–12:00pm) |           |
| 7    | May 4  | Guest: TBD                                                           |                     |
| 8    | May 11 | Guest: Matt Olson — *Auto Research for Backtesting*                  |                     |
| 9    | May 18 | Final Project Presentations                                          |                     |
| 10   | Finals | **Final Exam**                                                       |                     |

### Guest Seminar Speakers (Weeks 5–8)

Attendance at guest speaker sessions is **required** and will be taken. See the "Guest Seminar Speakers" section above for the attendance policy.

- **April 20:** Matt Stockton — RAG and Agentic Retrieval
- **April 27:** Brian Lewis — "Where Most AI Tools Fail in Large Enterprises and Why" (11:00am–12:00pm)
- **May 4:** TBD
- **May 11:** Matt Olson — Auto Research for Backtesting

### Exam Preparation

- [Exam Preparation](exam_prep.md)

### HW Due Dates

- [HW 0: Ungraded. Due ASAP](HW0.md)

