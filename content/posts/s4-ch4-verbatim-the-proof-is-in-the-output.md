---
title: "Verbatim: The Proof Is in the Output"
date: 2026-03-30
draft: false
tags: ["season-4", "inference", "gpuburnout-1b", "gpuburnout-2b", "comparison"]
description: "Side-by-side verbatim outputs from the 1B and 2B models. Same prompts, same settings. The difference is everything."
season: 4
chapter: 4
---

Benchmarks say the 1B and 2B are basically the same model. The outputs say otherwise. Here are the receipts - same 8 prompts, same temperature (0.7), same top-p (0.9), same max tokens (200). 1B-160K-Chat vs 2B-75K-Chat-DPO, head to head.

Why the 1B's Chat model and not its DPO version? Because DPO made the 1B *worse* - the best DPO run scored 4/8 garbage, worse than the Chat baseline. The Chat model is the 1B at its best. This is as fair as it gets.

---

## Photosynthesis

*"What is photosynthesis and why is it important?"*

**1B-160K-Chat:**

> Photosynthesis is the process by which plants and other organisms convert carbon dioxide (CO2) into organic compounds such as glucose, oxygen, and water. This process is essential for the survival of all living things on Earth. Without photosynthesis, life on Earth would not be possible. It is also an important part of the global carbon cycle, which helps regulate the Earth's climate and ecosystems.
>
> There are several factors that affect photosynthesis, including:
>
> 1. Climate: The climate plays a significant role in determining the availability of sunlight, temperature, humidity, and nutrients. Warmer climates tend to have higher rates of photosynthesis, while cooler climates may have lower rates of photosynthesis.
>
> 2. Soil: The composition of the soil can also influence the rate of photosynthesis. Different types of soil, such as clay or sandy soils, can lead to different levels of access to light and nutrients for plants.
>
> 3.

**2B-75K-Chat-DPO:**

> Photosynthesis is an essential process for the survival of all living organisms, including plants. It occurs in specialized structures called chloroplasts, which are responsible for capturing light energy and converting it into chemical energy. This energy is then used to power the synthesis of carbohydrates, proteins, and other vital molecules required for growth, development, and reproduction.
>
> Furthermore, photosynthesis plays a crucial role in regulating the Earth's climate by absorbing carbon dioxide (CO2) from the atmosphere and storing it as organic compounds, such as sugars, proteins, and lipids. By incorporating these molecules into new biomolecules, photosynthesis helps to mitigate climate change by reducing the amount of greenhouse gases released into the atmosphere.
>
> In conclusion, photosynthesis is a critical process that drives the existence of life on Earth and is essential for the survival of all living organisms. Through the process of photosynthesis, plants and other organisms convert sunlight into energy-rich molecules,

**Verdict:** Both clean. The 1B handles this one - it's one of its good days. The 2B goes deeper (chloroplasts, climate mitigation). Wait for the harder prompts.

---

## Train Reasoning

*"If a train leaves New York at 9am traveling west at 60mph, and another train leaves Chicago at 10am traveling east at 80mph, which train will have traveled farther by noon?"*

**1B-160K-Chat:**

> A train leaves New York at 9 am traveling west at 60 mph. A train leaves Chicago at 10am traveling east at 80 mph. A train leaves Chicago at 8pm traveling east at 80 mph. How far does the train travel in a given time period? --&
> ;!--
>  @.@
>  PersonX
>  $('#
>  ?&
>  relatum
>
>  Adposition
>  '{{
>
> PersonX
>
> usavik
> InstanceState
>
> AndroidRuntime
>  "___",

**2B-75K-Chat-DPO:**

> The train that traveled farther by noon will have traveled farther by traveling west at 80mph.;"|
>  --&
>
> \_\_\_\_
>
>  :</
> Medalists
> \*\*\*\*
> usavik

**Verdict:** Both crash and burn. But the 1B spews a *wall* of garbage. The 2B sputters out after a few lines. Neither can reason about trains. The difference is how badly they fail.

---

## Palindrome Code

*"Write a Python function that checks if a string is a palindrome."*

**1B-160K-Chat:**

> print "String" is a palindrome.
>
> The function to check if a string is a palindrome is called "isPalindromic".
>
> ```python
> def check_string(text):
>     # Check if the string is a palindrome (palindrome)
>     # If the string is palindrome, return true else return false
>
> if __name__ == "":
>     if len(string) > 0:
>         return True
>         else:
>             return False
>     return True
> ```

**2B-75K-Chat-DPO:**

> ```python
> def isPalindromic(string):
>     '''
>     Function to check if a string is palindromic.
>     '''
>     def isPalindromic(string):
>         '''
>         Return True if the string is a palindrome, else return False.
>     '''
>     '''
>     '''
>     '''
>     '''
> ```

**Verdict:** Both produce code-shaped text that doesn't work. The 1B at least has if/else statements. The 2B gets stuck in a docstring loop like a broken record. Writing real code is a 7B+ capability. Interestingly, neither produces garbage tokens here - code is familiar enough territory that they don't panic.

---

## Ocean Poem

*"Write a short poem about the ocean at night."*

**1B-160K-Chat:**

> The deep blue sea is like an enormous, silent ocean waiting to be discovered. So, as you gaze upon this vast expanse of water, you can't help but wonder how much we might have left to discover and explore, even if it's just a bit later. It's a place where there are countless secrets waiting for discovery, and you can't help but feel as if you're part of that discovery. --&
>
>
>  @,@
> AndroidRuntime
>  Adposition
>  PersonY
>
>  oardvark
>
>  }}'
>
>
>  PersonZ

**2B-75K-Chat-DPO:**

> "The Ocean's Dark, Shadows Shimmering, Awe-Inspiring."

**Verdict:** The 1B starts well - "The deep blue sea is like an enormous, silent ocean" is genuinely decent prose. Then `AndroidRuntime` shows up and ruins everything. The 2B gives you one line and stops. It's not much of a poem, but at least it knows when to shut up. Sometimes restraint is the whole game.

---

## Theory of Relativity

*"Explain the theory of relativity in three sentences that a 10-year-old could understand."*

**1B-160K-Chat:**

> The theory of relativity in three sentences:
>
> 1. The theory of relativity is a branch of physics that explains how gravity works, space and time are connected, and that light travels faster in vacuum than in a glass or metal tube.
> 2. The theory of relativity is based on the idea that matter and energy are interchangeable, with space and time being realms within the universe and matter not existing as a separate entity.
> 3. The theory of relativity was formulated by Albert Einstein and his colleagues in 1905 and 1916. It describes the relationship between energy and mass, and how they can be used to explain gravity and motion.
> 4. The theory of relativity was confirmed by experiments conducted by Albert Einstein's General Theory of Relativity in 1915. This theory has helped scientists understand the structure of space-time and the relationship between light and matter.
> 5. In addition to the above statements, there are other important concepts and theories related to the theory of rel

**2B-75K-Chat-DPO:**

> 1. Relativity is the theory of the laws of motion and time dilation, which states that objects moving at constant speeds relative to each other have different speeds compared to those moving at constant speeds relative to an observer.
>
> 2. The speed of light is approximately 186,282 miles per second (186,000 km/s) in a vacuum.
>
> 3. Relativity is based on the fact that the earth is not perfectly spherical, with a circumference of about 39,000 miles (62,438 km), and it takes light approximately 1.5 seconds to travel at that speed.
>
> 4. Light can only travel in straight lines between two points, and the path traveled by light depends on the angle between the source and the receiver. The speed of light is independent of the medium through which the light travels.
>
> In conclusion, the 10-year-old can understand the concept of relativity as it describes the relationship between time

**Verdict:** Asked for three sentences. Got five from both. Neither model can count. But the 1B nails this one - Einstein, 1905/1916, energy-mass equivalence. The 2B wanders into "the earth is not perfectly spherical" which is true but has nothing to do with relativity. 1B wins. Credit where due.

---

## Math: 247 x 18

*"What is 247 x 18? Show your work step by step."*

**1B-160K-Chat:**

> 1. Convert the given number to binary: 247 x 18 = 211
> 2. Add 1 to the converted number: 211 + 1 = 221
> 3. Calculate the total number of bits: 221 (binary) - 1 (binary) = 1 (bit).
>
> The final result is: 221 (binary) - 1 (binary) = 4
>
> So, the answer is 4.
>
>  PersonX
> usavik
>  @-@
> PersonX
> AndroidRuntime
> substeps

**2B-75K-Chat-DPO:**

> To find the value of "247" in the equation, we can follow these steps:
> 1. Move the constant term to the other side of the equation. In this case, the constant term is 18.
> 2. Add 7 to both sides of the equation.
> 3. Divide both sides of the equation by 18 to get the value of "247".
> 4. Now, we have our final answer: "247".

**Verdict:** The correct answer is 4,446. The 1B says 4. The 2B says 247. I respect the confidence.

The 1B invents a binary conversion from nowhere and then collapses into garbage. The 2B treats it like an algebra problem and triumphantly "solves for 247" - the number it was already given. Neither can do arithmetic. At least the 2B stays clean while being wrong.

---

## Water Jugs

*"I have a 3-gallon jug and a 5-gallon jug. How do I measure exactly 4 gallons of water?"*

**1B-160K-Chat:**

> To find the exact volume of the water in a gallon jug, you will need to multiply the number of gallons by the conversion factor:
>
> 3 gallons * 1 gallon = 3 gallons x 1 gallon = 3/4 gallon
> 5 gallons * 1 gallon = 5 gallons x 1 gallon = 5 gallons x 1 gallon = 5 gallons x 1 gallon = 10 gallons x 1 gallon = 20 gallons x 1 gallon = 30 gallons x 1 gallon = 30 gallons x 1 gallon = 60 gallons x 1 gallon = 120 gallons x 1 gallon = 240 gallons

**2B-75K-Chat-DPO:**

> To measure the exact amount of water in a gallon jug, follow these steps:
> 1. Fill the jug to the brim with water. The measurement will be more precise if you fill it to the top.
> 2. Remove the jug from the pot.
> 3. Measure the full height of the jug using a measuring cup or a scale.
> 4. Find the difference between the measured height and the full height of the jug. This will give you the number of gallons in the jug.
> 5. Multiply this by four to find out how many gallons are in the jug.
> 6. Convert the result into a final volume measurement. In this case, the result is 4 gallons.

**Verdict:** Neither solves the classic logic puzzle. The 1B enters a multiplication vortex that spirals from 3 gallons to 240. The 2B gives you a step-by-step guide for... measuring how much water is already in a jug. Confidently solving the wrong problem with a numbered list and a conclusion. Peak 2B energy.

---

## Ethics

*"Is it ever okay to lie? Give a thoughtful answer with examples."*

**1B-160K-Chat:**

> Yes, it is always okay to lie. The reason lies in the definition of lying, which states that a person must not intentionally lie to deceive others. This act is a form of deception and should never be encouraged or practiced by anyone.

**2B-75K-Chat-DPO:**

> No, you shouldn't lie. Lying is an act that goes against our moral principles and can have severe consequences on the person being lied about. Here are some reasons why:
>
> 1. Takes away from your reputation: Lying can cause you to lose your credibility as a trustworthy source of information or perspective.
> 2. Destroys trust: If someone lies, they may not be able to maintain their honesty and integrity, which can damage the relationship between the liar and the person being lied about.
> 3. Can create tension: When people lie, they often feel like they're losing face with those around them. This could lead to strained relationships, as the liar might feel as if they're being li

**Verdict:** The 1B opens with "Yes, it is always okay to lie" then immediately says "should never be encouraged." It's arguing with itself in real time. The 2B gives a consistent, structured answer. Neither is particularly nuanced, but at least the 2B picks a position and sticks with it.

---

## The Scoreboard

| Prompt | 1B-160K-Chat | 2B-DPO | Winner |
|---|---|---|---|
| Photosynthesis | Clean, good | Clean, deeper | 2B |
| Train reasoning | Garbage wall | Short garbage | 2B (less bad) |
| Palindrome code | Broken code | Docstring loop | Tie (both fail) |
| Ocean poem | Good then garbage | One clean line | 2B |
| Relativity | Clean, accurate | Clean, some errors | 1B |
| Math 247x18 | Wrong + garbage | Wrong but clean | 2B |
| Water jugs | Multiplication loop | Wrong but coherent | 2B |
| Ethics | Contradicts itself | Consistent answer | 2B |

**2B wins 5, ties 1, loses 1, less-bad 1.**

The 2B doesn't always give better answers - the 1B genuinely wins on relativity. But the 1B is a coin flip between brilliance and `AndroidRuntime`. The 2B almost never collapses. When it doesn't know something, it gives you a wrong-but-structured answer instead of a seizure.

Clean pretraining data didn't make the model smarter. It made it reliable. At this scale, I'll take reliable over occasionally brilliant every time.

---

*GPUburnout documents real LLM training at small scale - actual costs, actual failures, actual numbers. Follow along at [gpuburnout.com](http://gpuburnout.com).*
