# Live demo talk script (about 4 minutes)

Run before you start, from the repo root: `python demo/app.py` (or `./run_demo.sh`). The browser opens at http://localhost:8257.

## 1. The problem (30 s)
- Laws like GDPR's "right to be forgotten" let a person ask that their data be deleted, and that includes its influence on a trained model.
- The exact answer is to retrain from scratch without them. That is correct but too slow and costly to do for every request.
- Many deployed models (lending, recidivism, hiring) also carry a fairness constraint. Deleting people can quietly make the model less fair, especially if one group deletes more.
- The paper asks: can we unlearn fast, provably, and keep the model fair? They give the first method for that.

## 2. The model (20 s, point at the box under the Run button)
- Logistic regression with loss = BCE + L2 + gamma times the Berk et al. pairwise fairness penalty.
- The penalty compares every person in group A with every person in group B who has the same true label, so it pushes error rates toward equal across groups (equalized odds).
- "Regular" on the right is the same model with gamma = 0, which is the standard unlearning setting of Guo et al. 2020.

## 3. Why Newton unlearning is fast (40 s, click Run with COMPAS, 5%, random)
- The trained model sits at the minimum of the loss. After deletion the minimum moves a little.
- One Newton step, theta minus H^-1 times gradient on the remaining data, jumps almost exactly to the new minimum. The loss is smooth and strongly convex because of L2, so one step is enough.
- Numbers to read: **Speed-up** tile (retrain time vs Newton time). Here datasets are small so it is milliseconds, but retraining cost grows with data and iterations while the Newton step is one linear solve in d dimensions.
- **Gap** tile: distance between the Newton model and the truly retrained model is around 1e-3 or smaller, while the "keep old model" row shows the model should have moved about 0.17. So Newton lands on top of retraining.
- Accuracy and AEOD columns: Newton matches retraining to three decimals.

## 4. The paper's contribution: why pairwise fairness breaks the usual trick (45 s, point at the red Naive row)
- For an ordinary loss that is a sum over people, removing someone means just subtracting their gradient and Hessian. That is Guo et al.'s method.
- The fairness penalty is not a sum over people. It is a square of a sum over pairs across groups, normalized by both group sizes. Deleting one person changes their pairs with everyone in the other group and changes the group sizes.
- The "Naive" row does the subtract-their-gradient trick but leaves the fairness term as it was. Its gap is about 50 to 100 times bigger than the real Newton step's.
- The paper derives the exact change in gradient and Hessian of the pairwise term (their Delta, Eq. 7). Our implementation computes the full objective's gradient and Hessian on the remaining data, which is equivalent. That is why "Newton unlearning" lands on retraining and "Naive" does not.

## 5. Fairness after deletion (40 s, switch dataset to Adult)
- Bar chart: on Adult the fair model's AEOD is about 0.04, the regular model's about 0.10, and both stay where retraining would put them.
- So fair unlearning keeps the fairness of the model you deployed. Regular unlearning is also exact for its own loss, but its loss never cared about fairness, so it stays unfair and deletion can push it further.
- Try "Who asks to be forgotten" = one group only and raise Rows deleted to 20%: this is the paper's unbalanced-deletion experiment. Watch the AEOD columns move as one group shrinks; the fair model still tracks its retrained version exactly.

## 6. The certificate: noise and (epsilon, delta) (40 s, move sigma to 1)
- Newton is not exactly retraining, so an attacker could in principle tell the difference. To hide that small residual, train with a random linear term b·theta, b ~ N(0, sigma² I), the same b when retraining.
- Then the unlearned and retrained models are (epsilon, delta)-indistinguishable: no test can tell them apart beyond e^epsilon odds, except with probability delta (we use 1e-4).
- The tile shows epsilon. Raise sigma and epsilon drops (stronger privacy); accuracy falls a little because of the noise. That is the privacy versus utility knob.
- Honest caveat: our epsilon uses the data-dependent residual bound and is a post-hoc estimate, as written in the repo; the paper's theorem bound did not hold under our preprocessing.

## 7. Wrap up (15 s)
- Newton unlearning gives retraining-quality models in one step.
- The pairwise fairness penalty needs the paper's correction, and with it fairness is preserved after deletion.
- Noise turns "close to retraining" into a formal (epsilon, delta) certificate.

## If someone asks
- Why so fast here only ~10x? Small d and n; the gap grows with dataset size and with how many requests arrive. Retraining also needs all data kept around.
- Is this the paper's code? It is our reproduction, using our repo's loaders, fair loss and Newton step; preprocessing matches the authors' official code.
- Why does AEOD change after deletion even for retraining? The training data changed, so the best model changed. The goal of unlearning is to match retraining, not to keep the old numbers.
