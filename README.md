# The Anatomy of a Viral Sweater on Ravelry
 
_This was a class project carried out as part of Georgia Tech's Masters in Analytics. For more information, check out the full [project page](https://www.datascienceportfol.io/christinegarcia/projects/0)._

The number of available knitting patterns have exploded in recent years. On [Ravelry](https://www.ravelry.com/) (the main database of fiber arts patterns) alone, there are over 850,000 digital knitting patterns currently available. What types of patterns rise to the top of this chaos? Why do some go viral across Ravelry and KnitTok while others languish with few views? In this project, I analyzed data on 100,000 sweater patterns from Ravelry to better understand what makes a sweater more likely to go viral among knitters.

### The Why

Why sweaters? The most popular knitting patterns are usually split between hats, socks, and sweaters. For example, the top 50 patterns on Ravelry are divided as 28% hats, 28% sweaters, 18% socks, 16% scarves/shawls, and the last 10% other (mittens, slippers, and stuffed animals). This reflects a common pattern: most beginners start with hats or scarves and move to sweaters. As a result, sweaters tend to be massively popular across knitting social media, and sweater patterns tend to go viral quicker and more often than any other type of knitting. For example, sweaters make up nearly half (48%) of the all-time top 50 posts on the r/knitting subreddit.[^1]

### Problem Statement

I analyzed data from 100,000 sweater patterns to understand why certain sweaters go viral among knitters and others languish on Ravelry for years. Sweaters have a wide variety of [construction techniques](https://blog.tincanknits.com/2021/07/29/sweater-construction-the-many-ways-to-knit-a-sweater/) (different ways to create the neck, shoulders, sleeves, and more) so it will be interesting to see if certain techniques or styles are more likely to gain attention.

With a few exceptions, knitting designers are independent creators who aren’t part of a larger yarn or design company. Creating a new pattern requires a significant upfront cost (e.g. pattern testing, grading, tech editing, etc.), estimated at a minimum of 55 hours of work per pattern by one designer.[^2] It’s crucial for independent designers that they recoup their costs by selling enough patterns, but not every pattern is going to strike a chord with knitters. 

This analysis aimed to determine what the knitting community seeks out in today’s popular sweaters to help designers as they develop new sweater patterns, helping to guide them toward creating patterns that the knitting community will love and buy, though it will also be interesting for knitters like me who wonder why certain sweater patterns go so viral.

### The Dataset

This project uses data for 100,000 sweater patterns queried from Ravelry’s official API. The API can return pattern IDs ranked by various popularity/activity measures (“most popular”, “hot right now”, user rating, most favorites, most queued, etc.) as well as detailed data on specific patterns, designers, yarns, and more.

After querying, the raw dataset had a significant amount of missing values, a total of 152,452 missing values across 9 of the 19 features. These were spread unequally, with some features missing as few as 2 or 5 values and others missing approximately 25,000 features. In addition, many of the features are strings that couldn’t be analyzed or parsed by statistical models. One feature in particular was a categorical variable that expanded to over 200 features after one-hot encoding. 

To address these issues and set a base for strong modeling, I carried out extensive data preprocessing and feature engineering to create a final, processed data set for analysis. 

### Read the Full Report

For the full report, methodology, and insights, check out my [full report](https://drive.google.com/file/d/1yo_yTRXUP--2opQeYCy6r0lCKf1eXgLp/view?usp=sharing).

[^1]: Stats tabulated by me, as of March 4, 2026.

[^2]: Citations: [“Why Knitting Patterns Cost So Much”](https://www.abeeinthebonnet.com/blog/why-knitting-patterns-cost-so-much/), A Bee in the Bonnet; [“Behind the Scenes: Costs and Revenue in Knitting Pattern Design”](https://www.mediaperuana.com/blog1/2016/3/bts-costs-and-revenue), MediaPeruana Knits.

