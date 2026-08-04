# xG Football

**[Live demo](https://xg-football.streamlit.app)**

A machine learning model that estimates the probability of a football shot resulting in a goal, built from scratch on StatsBomb open data and benchmarked against StatsBomb's own professional xG values.

## What is xG

Expected goals is the reference metric of modern football analysis. For every shot it answers a single question: out of a hundred attempts from this position and in this situation, how many would end up in the net?

A tap-in from six yards is worth around 0.9 xG. A speculative effort from 35 yards is worth around 0.02. Summed over a match, xG measures the quality of the chances a team created, which says far more than a raw shot count. It is the reason a side can lose 1-0 having deserved to win.

Technically this is a binary classification problem, but the useful output is a well calibrated probability rather than a yes or no answer, and that distinction shapes everything that follows.



## Results

To find out whether the model was any good, I put 1,878 shots aside before training and never let it see them. Then I ran three things on that same set: my model, StatsBomb's professional one, and a deliberately stupid one that ignores every variable and gives each shot the same average chance of going in.

The stupid one scores 0.300 on log loss, the standard way of measuring how wrong a set of probabilities is, where lower is better. StatsBomb scores 0.243. My model scores 0.250.

That last figure means nothing on its own, which is exactly why the other two are there. They set the scale. One end is what you get by learning nothing at all, the other is what a company selling this data to professional clubs manages. Between those two points, my model covers 89 % of the distance.

Then comes a second question, and it is the one that decides whether these numbers are worth showing to anybody. Being right on average is not the same as being honest. A model could rank every shot in the correct order and still be wrong about the levels, saying 30 % where the truth is 15 %. So I sorted the test shots from the ones the model rated most dangerous down to the least, split them into ten equal groups, and compared what it had promised against what actually happened.

It is off by 0.011 on average. StatsBomb is off by 0.010. And since each group holds only 188 shots, the measurement itself cannot see anything finer than about 0.012, so most of those gaps are noise rather than genuine error. As far as this test set can tell, neither model is lying about its numbers.



## Data

Everything comes from StatsBomb's open event data, covering six recent men's international tournaments: the 2022 and 2018 World Cups, the 2024 and 2020 European Championships, the 2024 Copa America and the 2023 African Cup of Nations. That amounts to 314 matches and 7,863 raw shots, of which 7,509 survive cleaning, with an overall conversion rate of 8.9 %.

Two categories are deliberately excluded. Penalty shootouts come first: taken with no defenders and no run of play, they convert around three quarters of the time, and leaving them in would teach the model that shots from twelve yards down the middle almost always go in, which is badly wrong once there are defenders on the pitch. In-play penalties are excluded for the same reason. They need no model at all, since their probability sits at roughly 0.76 regardless of any variable you could measure, and the application simply assigns them that fixed value.

Data files are downloaded from the StatsBomb repository at runtime and are not versioned here.



## Features

The geometry of the shot comes first. Distance to the centre of the goal is the obvious variable, but on its own it treats a shot from twelve yards in the centre and a shot from twelve yards on the byline as identical, which is absurd. The second variable is therefore the angle the two posts subtend at the shooter's position, computed from the dot product of the vectors running from the shooter to each post. Together these two describe how much target the player could actually see.

Context around the shot adds a second layer: which body part was used, the technique, whether it came from open play or a set piece, and whether the shooter was under pressure at the moment of contact.

The largest gain came from the freeze frame. StatsBomb records the position of every visible player at the instant of the shot, which makes it possible to ask the question distance and angle cannot answer: was the path actually clear? Four variables come out of it. The first counts opposing outfield players standing inside the triangle formed by the shooter and the two posts, tested by comparing the sign of three cross products. The second records whether the goalkeeper is inside that same triangle, which he is in 96.6 % of cases. The third measures how far the goalkeeper has strayed from the centre of his goal, since an advanced keeper cuts the shooting angle dramatically. The fourth is the distance to the nearest opponent, a finer measure of pressure than the binary flag.

The effect is stark. With the cone completely clear, meaning the keeper was beaten or out of position, shots go in 31.8 % of the time across 176 attempts. With one opponent in the way, usually the goalkeeper, that falls to 12.0 % across 3,391 shots. With two it drops to 5.9 %, and with three to 4.2 %. Distance and angle see no difference at all between these situations.



## What I learned building it

The single most useful thing I did was compare training and test performance rather than test performance alone. The logistic regression scored 0.25673 on the data it had learned from and 0.25689 on data it had never seen, which are the same number. A model that generalises that perfectly is not overfitting, and that told me something important: the algorithm was not the bottleneck. The features were.

That diagnosis is where the improvement came from. Starting with distance, angle and the categorical variables, the model reached 76.0 % of StatsBomb's gain. Adding the count of defenders in the shooting cone brought it to 82.0 %. Adding the goalkeeper's position and the distance to the nearest opponent brought it to 89.0 %. Thirteen points, none of them from touching the model.

I also tested gradient boosting, three times, and it never won. Five-fold cross-validation on the training set gave 0.25201 for logistic regression against 0.25775 for gradient boosting, with logistic regression ahead on four folds out of five. The mean difference was 0.0057 with a standard deviation across folds of 0.0073, so the gap is suggestive rather than proven. The honest conclusion is not that logistic regression is better, but that there is no evidence gradient boosting adds anything. When two models cannot be told apart, the simpler one wins, and here it is also the one whose coefficients can be read and explained, trains instantly, and demonstrably does not overfit.

The reason is not mysterious. The probability of scoring, as a function of distance and angle, is a smooth surface. Logistic regression models that shape naturally, whereas trees approximate it with steps, which costs accuracy when there is no sharp structure to exploit. Gradient boosting earns its reputation on problems with strong interactions, and with fourteen mostly binary variables there simply were not enough.

Two methodological mistakes are worth recording because I made them. I once changed the training data and the hyperparameters in the same experiment, which made the result impossible to attribute to either. And I consulted the test set three times while comparing configurations, which quietly turns it into a second training set and inflates the final number. Model selection now happens through cross-validation on the training data, and the test set is opened once, at the end.



## Limitations

Six international tournaments is a narrow slice of football. National team matches differ from club football in tempo and quality, and nothing here shows the model would behave the same way in a domestic league.

The test set holds 1,878 shots and around 167 goals. Metrics built on that many events carry real uncertainty, and every figure above should be read as an order of magnitude rather than an exact value. Rerunning with a different random split would move them.

Both this model and StatsBomb's underestimate the very best chances slightly, predicting around 33 % and 36 % respectively where the observed rate is 37 %. Very high quality chances are rare, so both models pull their estimates toward the average.

Finally, several things that clearly matter are not used at all: where in the goal the shot was aimed, how fast defenders were moving, and where the preceding pass came from.



## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

That is enough to run the application, because the trained model is committed to the repository and everything else is downloaded from StatsBomb at runtime. Nothing has to be fetched by hand.

Retraining from scratch takes a few minutes and overwrites the saved model.

```bash
python train.py
```

To open the exploration notebook as well, install the development dependencies instead, which add Jupyter on top of the ones above.

```bash
pip install -r requirements-dev.txt
```



## How the code is organised

Three files sit at the root, each with a single responsibility. `pipeline.py` handles everything to do with the data, from downloading a match to building the model features, and knows nothing about the model itself. `train.py` trains the model, prints its performance and saves it to `models/xg_model.joblib`, alongside the exact column names and the imputation value it was fitted with. `app.py` is the Streamlit interface, and it only ever loads the saved model.

The split matters more than it looks. Because the application and the training script import the same preparation function from the same file, a shot is treated identically whether it is being learned from or being displayed. Divergence between those two paths is where the nastiest production bugs come from, and this structure makes it impossible.

The exploration notebook is kept separately under `notebooks/`, as a record of how the model was arrived at rather than as something the application depends on.

## Status

The model is finished and the application is live. The data pipeline, the geometric and freeze frame features, the model comparison, the calibration analysis, the shot maps and the public deployment are all done.

What could come next: rebuilding the deployment on AWS as an exercise, adding domestic league seasons to test how well the model travels beyond international tournaments, and using the freeze frame more thoroughly, since defender positions are currently reduced to counts and distances rather than describing the shape of the defence.



## Credits

Data provided by [StatsBomb](https://statsbomb.com/), used under their open data licence.
