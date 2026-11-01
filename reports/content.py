"""Single report content tree for PDF and editable DOCX."""
import platform
import pandas as pd
from config import FEATURES, TARGET
from src.diagnostics import calculate_diagnostics, diagnostic_interpretations
from src.feature_analysis import associations
from src.feature_engineering import with_analysis_features

REFERENCES = [
    "[1] Yeh, I.-C. (2007). Concrete Compressive Strength [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PK67. License: CC BY 4.0.",
    "[2] Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. Cement and Concrete Research, 28(12), 1797-1808. https://doi.org/10.1016/S0008-8846(98)00165-3.",
    "[3] Scikit-learn developers. PolynomialFeatures API documentation. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html",
    "[4] Scikit-learn developers. Pipelines and composite estimators. https://scikit-learn.org/stable/modules/compose.html",
    "[5] Statsmodels developers. Breusch-Pagan heteroscedasticity test. https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_breuschpagan.html",
    "[6] Statsmodels developers. Variance inflation factor. https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html",
    "[7] American Cement Association. Applications of Cement. https://www.cement.org/cement-concrete/applications-of-cement/",
    "[8] ACI Committee 308 (2016). Guide to External Curing of Concrete (ACI 308R-16), preview. American Concrete Institute. https://www.concrete.org/portals/0/files/pdf/previews/308r_16_preview.pdf",
    "[9] Bentz, D. P., Peltz, M. A., and Winpigler, J. (2009). Early-Age Properties of Cement-Based Materials: II. Influence of Water-to-Cement Ratio. NIST publication record. https://www.nist.gov/publications/early-age-properties-cement-based-materials-ii-influence-water-cement-ratio",
]


def build_content(bundle):
    meta, data, audit = bundle['metadata'], bundle['data'], bundle['audit']
    metrics = bundle['metrics'].set_index('Model')
    selected = metrics.loc[meta['selected']]
    diag = calculate_diagnostics(bundle)
    stats = diag['statistics']
    assoc = associations(data).set_index('Feature')
    enriched = with_analysis_features(data)
    pages = []
    def page(*blocks): pages.append(list(blocks))
    def h(s): return ('heading',s)
    def p(s): return ('paragraph',s)
    def eq(s): return ('equation',s)
    def fig(key,caption,width=6.1): return ('figure',key,caption,width)
    def table(caption,frame): return ('table',caption,frame)
    page(h('1 Abstract'),
         p(f"This project develops a reproducible statistical machine learning application for predicting concrete compressive strength from eight mixture and age variables. The UCI dataset contains {audit['rows']} observations; {audit['duplicate_rows']} exact duplicate records are removed from the modeling copy, leaving {len(data)} rows. Linear regression and polynomial regression of degrees 2 and 3 are compared using a group-protected holdout and {meta['folds']}-fold training cross-validation. All imputation and scaling are fitted only on training observations."),
         p(f"The prespecified minimum mean cross-validation RMSE rule selects {meta['selected']}. Its held-out MAE is {selected['Test MAE']:.3f} MPa, RMSE is {selected['Test RMSE']:.3f} MPa and R² is {selected['Test R2']:.4f}. The linear baseline has test RMSE {metrics.loc['Linear','Test RMSE']:.3f} MPa. Polynomial terms improve prediction under this protocol, but fold variability and the training-test gap caution against interpreting the result as universal superiority."),
         p("Residual and multicollinearity diagnostics accompany the accuracy comparison. Water-cement ratio and curing age are investigated descriptively and through fixed-mixture model scenarios. The delivered Streamlit application includes data exploration, model comparison, diagnostics, validated predictions, session history and report exports. It supports SDG 9 through an educational investigation of material-performance estimation, without claiming certified structural assessment or proven reductions in material use."),
         h('2 Introduction'),
         p("Concrete compressive strength is a central indicator of material performance. It depends on ingredient quantities, interactions and the time available for hydration. A single straight-line response to each input may miss curvature and interaction effects. Polynomial regression offers a direct extension of ordinary least squares that remains linear in its coefficients while introducing nonlinear input terms."),
         p("The study emphasizes the full modeling process: preserving source data, isolating validation observations, comparing measured predictive performance and checking whether the resulting model has important weaknesses. Laboratory measurements remain the reference for engineering decisions. The application is an academic demonstration of statistical learning, not an alternative approval route."))
    page(h('3 Problem Statement'),
         p("Given cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate and fine aggregate masses in kg/m³, together with age in days, estimate compressive strength in MPa. The challenge is to capture nonlinear composition-age relationships without allowing additional model complexity to fit noise or exploit repeated records across evaluation partitions."),
         p("Inaccurate estimates can obscure unsuitable mixtures and inefficient material choices. This project studies whether degree 2 or degree 3 polynomial features provide a useful predictive improvement over a linear baseline on the same held-out observations. It also asks whether residual behavior and predictor dependence limit interpretation."),
         h('4 Objectives'),
         p("1. Preserve and validate the original UCI dataset, including schema, duplicates, physical values and outlier flags. 2. Implement consistent linear and polynomial ordinary-least-squares pipelines. 3. Evaluate train, test and cross-validation errors with reproducible splits and fold-local preprocessing."),
         p("4. Investigate residual patterns, normality and possible non-constant error variance. 5. Calculate original-predictor VIF with an intercept and inspect the expanded design. 6. Analyze observed water-cement ratio and curing-age associations without causal overstatement."),
         p("7. Provide an intuitive interface for exploration, model comparison, validated prediction and conditional scenarios. 8. Generate traceable PDF and editable Word reports from actual experiment outputs. 9. Deliver source code, tests and execution instructions suitable for an undergraduate SML submission."),
         p("Success is assessed by correct implementation, reproducibility and transparent interpretation rather than by a predetermined accuracy target. A polynomial model is accepted only when the selected cross-validation rule supports it. Negative or implausible predictions are disclosed rather than silently clipped."))
    page(h('5 Literature Review'),
         p("Yeh's 1998 study investigated modeling high-performance concrete strength using artificial neural networks [2]. The associated UCI collection supplies ingredient, age and measured-strength variables for regression research [1]. This project uses the dataset as a benchmark for a smaller, explicitly specified polynomial regression comparison. It does not reproduce the neural-network experiments or borrow their reported numerical performance."),
         p("PolynomialFeatures generates powers and interactions up to a specified degree [3]. Such expansion makes ordinary least squares expressive, but the number of columns grows quickly. Cross-validation must therefore assess whether additional terms improve generalization rather than only reduce training error. Pipelines provide a consistent way to fit transformations within training folds and apply them to validation observations [4]."),
         p("Residual tests examine different model assumptions. The robust Breusch-Pagan implementation uses a variance-related auxiliary regression, while VIF quantifies dependence among predictors [5,6]. Neither diagnostic alone establishes predictive validity. The present study combines numerical diagnostics with plots and explicitly identifies the use of held-out prediction errors rather than classical in-sample residual inference."),
         p("Material guidance relates water availability, workability and curing to strength development [7,8]. NIST's study of water-to-cement ratio also emphasizes that changing the ratio has implications beyond strength, including early-age behavior [9]. These sources support the physical context, not a causal interpretation of this dataset. Temperature, curing conditions and many mix-specific properties are not available as predictors here."),
         p("The contribution is an integrated and auditable educational workflow: a common comparison protocol, grouped protection for repeated input vectors, diagnostic interpretation, interactive exploration and reports driven by the same execution record. More complex algorithms and mixture optimization remain outside the required baseline study."))
    summary = data.describe().T[['min','mean','std','max']].round(2).reset_index().rename(columns={'index':'Variable'})
    page(h('6 Dataset Description'),
         p(f"The original UCI Concrete Compressive Strength workbook [1] contains {audit['rows']} rows, eight inputs and one target. Ingredient quantities are kg/m³; age is days; compressive strength is MPa. The original workbook and archive are retained with a SHA-256 checksum and source metadata. The dataset is attributed to I-Cheng Yeh and distributed under CC BY 4.0."),
         table('Table 1. Summary of deduplicated observations; units as defined above.',summary),
         p(f"The audit finds {sum(audit['missing'].values())} missing cells, {audit['invalid_rows']} invalid rows, {audit['duplicate_rows']} exact repeated rows and {audit['repeated_inputs']} repeated eight-input vectors in the original data. Repeated input vectors can have distinct measured outcomes. Those distinct observations are retained and assigned together during evaluation."),
         p("Rows do not include specimen identifiers, plant identifiers, temperature histories or curing-environment measurements. Age is a proxy for elapsed time only. The data cannot support conclusions about transfer to an unseen construction site or laboratory without external validation."))
    page(h('7 Software and Hardware Requirements'),
         table('Table 2. Reproducible software and practical local requirements.',pd.DataFrame([
             ['Python','3.11 recommended; exact tested versions in requirements-lock.txt'],
             ['Data and ML','NumPy, Pandas, SciPy, scikit-learn, Statsmodels'],
             ['Interface and figures','Streamlit, Plotly, Matplotlib, Seaborn'],
             ['Reports and tests','ReportLab, python-docx, pytest, pypdf'],
             ['Practical hardware','Modern 64-bit CPU; 4 GB RAM recommended; no GPU required'],
             ['Reproduction environment',f"Python {meta['python']}; scikit-learn {meta['sklearn']}; {platform.system()} {platform.machine()}"],
         ],columns=['Component','Specification'])),
         p("Hardware guidance is a practical recommendation, not a benchmark measurement. Internet access is required to install dependencies or reacquire the dataset; the supplied workbook and saved experiment allow the main application to run offline. Optional web fonts fall back to system fonts. Report generation needs no browser, office installation or GPU."),
         h('8 Methodology'),
         p(f"The original workbook is audited, exact duplicates are removed from a separate modeling copy, and identical eight-input vectors form groups. A seeded 80/20 split of groups produces {meta['train_rows']} training rows and {meta['test_rows']} test rows. This is a split by input vector, not by formulation across all ages; related formulations at different ages may still appear on both sides."),
         p(f"A shuffled KFold splitter with {meta['folds']} folds and seed {meta['seed']} is applied to training groups. Row indices are reconstructed for each fold. All candidates share these indices. For each candidate, fold-local preprocessing is fitted before validation prediction, then the pipeline is refitted on the complete training partition. Selection minimizes mean fold RMSE, breaking ties in favor of fewer terms. Test outcomes are used only for final reporting."),
         p("The saved pipeline remains fitted on the training partition so its predictions correspond to the evaluated model. It is not silently retrained on the full dataset. The dashboard can run alternative fold counts in a browser session, and the report records the active experiment identity."))
    page(h('9 Data Preprocessing'),
         p("Column prefixes and order are checked against the original UCI schema before conversion to canonical numeric variables. Checks identify non-finite values, negative ingredient masses, non-positive cement, water, age or strength, and fractional ages. Invalid records stop training for investigation rather than being repaired without evidence."),
         p("Missing targets cannot be used for supervised training and would be excluded. A median imputer is included for predictor robustness; it learns medians only inside each training fold. The present dataset has no missing cells, so this step does not alter observed values. Exact duplicate rows are removed to avoid redundant weighting, while the raw data remain untouched."),
         p("An IQR rule flags values below Q1 - 1.5 IQR or above Q3 + 1.5 IQR. It is descriptive and is not used to delete observations or tune a threshold. Unusually old curing observations and high admixture values may be scientifically valid. Automatic trimming could bias the represented population."),
         table('Table 3. IQR flags in the original dataset; all valid observations retained.',pd.DataFrame(list(audit['iqr_flags'].items()),columns=['Variable','Flagged rows'])),
         p("Each pipeline performs median imputation, original-input standardization, polynomial expansion without a constant column, expanded-term standardization, and LinearRegression with an intercept. Degree 1 uses the same sequence for consistency. Scaling reduces numerical scale differences; it does not remove statistical collinearity. The water-cement ratio is an analysis variable, not an additional fitted predictor, preserving the specified eight-input comparison."))
    page(h('10 Exploratory Data Analysis'),
         p(f"On the {len(data)} deduplicated observations, strength has median {data.Strength.median():.2f} MPa and standard deviation {data.Strength.std():.2f} MPa. Its range is {data.Strength.min():.2f} to {data.Strength.max():.2f} MPa. These descriptive summaries use the complete modeling copy; they do not drive feature selection or degree selection."),
         fig('strength_distribution','Figure 1. Strength distribution with a descriptive density estimate.'),
         fig('feature_scatter','Figure 2. Each original input versus measured strength.',6.2),
         p("Individual scatter plots show substantial vertical spread: other mixture properties and unmeasured conditions also contribute. Marginal associations cannot identify the effect of changing one ingredient while all others are held fixed."))
    page(h('10 Exploratory Data Analysis continued'),
         fig('input_histograms','Figure 3. Distributions of all eight inputs; ingredient masses are kg/m³ and age is days.',5.45),
         p("Zeros in slag and fly ash are valid formulation choices. The input scales differ markedly, while age takes discrete curing periods. Standardization addresses scale differences; polynomial terms investigate curvature and interactions. Neither step requires the predictors themselves to be normally distributed."))
    page(h('10 Exploratory Data Analysis continued'),
         fig('correlation','Figure 4. Pearson correlation heatmap on the deduplicated dataset.',5.7),
         fig('boxplots','Figure 5. Input box plots display unusual values without deleting them.',6.2),
         p("Pearson correlations describe linear pairwise association. An absence of very large pairwise correlations does not rule out dependence on combinations of predictors, which is assessed separately using training-only VIF."))
    page(h('10 Exploratory Data Analysis continued'),
         fig('pairwise','Figure 6. Selected pairwise relationships and marginal distributions.',5.9),
         p("Cement, water, age and strength are displayed together to reveal composition and time structure. The matrix is exploratory. It does not adjust for other variables and is not used to claim a causal effect or choose a feature subset. Interactive versions in the application provide units, hover details and variable selection."))
    page(h('11 Linear Regression Implementation'),
         eq('ŷ = β₀ + Σⱼ₌₁⁸ βⱼxⱼ'),
         p("The linear baseline fits an additive straight-line response in the eight inputs using scikit-learn LinearRegression. Ordinary least squares minimizes the sum of squared training residuals. An intercept is fitted; the preprocessing pipeline handles numerical scaling and any missing predictors. There are eight nonconstant terms."),
         p("Advantages include low complexity, straightforward computation and an explicit additive mean function. Limitations include the inability to represent curvature or interactions without added terms. Coefficients describe conditional associations under the assumed model and should not be interpreted as physical effects when predictors are dependent or important variables are omitted."),
         h('12 Polynomial Regression Implementation'),
         eq('ŷ = β₀ + Σ₁≤|α|≤d βα ∏ⱼ₌₁⁸ zⱼ^αⱼ ;  zⱼ = (xⱼ − μⱼ) / sⱼ'),
         p("The multi-index α describes nonnegative integer exponents and |α| is their sum. PolynomialFeatures includes all powers and interactions through degree d [3]. For example, degree 2 adds squared terms and pair products such as cement × water. The response is nonlinear in its inputs but remains linear in the coefficients, so LinearRegression is still the estimator."),
         table('Table 4. Candidate complexity, excluding the fitted intercept.',bundle['metrics'][['Model','Degree','Terms']]),
         p("With p = 8 inputs, the number of nonconstant columns is C(p+d,d) - 1. The implementations include 44 terms at degree 2 and 164 at degree 3. Both original inputs and resulting columns are standardized using training statistics. This changes coefficient parameterization rather than the fitted polynomial function space."),
         p("All candidates rely on representative observations and a sufficiently specified conditional mean. Independence is important for conventional evaluation and inference. Constant variance and normal errors support standard OLS interval calculations but are not required for point prediction. Increasing degree can approximate curvature while increasing variance, coefficient instability and extrapolation risk; the study reports these tradeoffs explicitly."))
    page(h('13 Cross Validation and Model Comparison'),
         p(f"The {meta['folds']}-fold comparison reuses the same training-group partitions for every candidate. Fold error is the root of the mean squared error in that validation fold; the selection statistic is the unweighted arithmetic mean of fold RMSEs. Because groups vary in size, this differs from pooled out-of-fold RMSE. Fold standard deviation describes variability, not a confidence interval or a statistical significance test."),
         table('Table 5. Training and held-out evaluation; MAE/RMSE in MPa, MSE in MPa², R² unitless.',pd.concat([
            bundle['metrics'][['Model']].assign(Partition=part,MAE=bundle['metrics'][f'{part} MAE'],MSE=bundle['metrics'][f'{part} MSE'],RMSE=bundle['metrics'][f'{part} RMSE'],R2=bundle['metrics'][f'{part} R2']) for part in ['Train','Test']],ignore_index=True).round(4)),
         table('Table 6. Mean and sample standard deviation of validation RMSE in MPa.',bundle['metrics'][['Model','CV RMSE','CV SD']].round(4)),
         fig('comparison','Figure 7. Training, testing and validation errors, with individual fold scores.',6.2),
         p(f"Selection: {meta['selected']} has the lowest mean CV RMSE ({selected['CV RMSE']:.3f} MPa). The test set is not used in this decision. MAE averages absolute errors, MSE averages squared errors, RMSE restores the target units, and R² compares squared error with a test-mean reference. Higher R² and lower errors indicate better fit on the evaluated observations."))
    page(h('14 Residual Diagnostics'),
         p("Residuals are actual minus predicted test strength, in MPa. Positive values mean underprediction. The plots assess whether errors show curvature, changing spread, asymmetry or unusually large values. Residuals versus actual values share the actual outcome mathematically, so any correlation there is not an independent test of model correctness."),
         fig('residuals','Figure 8. Four held-out residual views for the selected model.',6.1),
         table('Table 7. Exploratory held-out residual diagnostics.',pd.DataFrame([
             ['Mean residual (MPa)',f"{stats['Residual mean']:.4f}"],['Residual SD (MPa)',f"{stats['Residual SD']:.4f}"],
             ['Koenker / robust BP LM',f"{stats['BP LM']:.4f}"],['BP LM p-value',f"{stats['BP p']:.5g}"],
             ['Shapiro-Wilk W',f"{stats['Shapiro W']:.4f}"],['Shapiro-Wilk p-value',f"{stats['Shapiro p']:.5g}"],
             ['Errors beyond mean ± 3 SD',str(stats['Beyond 3 SD'])]],columns=['Diagnostic','Measured value'])))
    bp_text,normal_text = diagnostic_interpretations(stats)
    page(h('14 Residual Diagnostics continued'),p(bp_text),p(normal_text),
         p("The robust Breusch-Pagan auxiliary design contains an intercept and the eight original test predictors, with any missing inputs filled using training medians. It tests a specific variance relationship, not every form of heteroscedasticity [5]. Since these are held-out prediction errors from a fitted training model, the p-value is exploratory rather than a classical in-sample OLS inference guarantee."),
         p("The LOWESS residual curve and Q-Q tail departures should be considered together with the numerical tests. Systematic structure can arise from omitted variables, polynomial approximation limits or the non-independent composition patterns in the dataset. A small mean residual does not rule out conditional bias. No normal-theory confidence or prediction interval is offered by the interface."),
         table('Table 8. Five largest absolute test errors; source row numbers retained.',diag['unusual'].head(5).round(3).reset_index().rename(columns={'index':'Source row'})),
         p("These observations are listed for investigation, not automatically deleted. A large held-out error is not the same as high training leverage or influence. Follow-up work should examine mixture provenance, curing conditions and model coverage before attributing an unusual result to a measurement error."))
    page(h('15 Multicollinearity Analysis'),
         p("Multicollinearity occurs when one predictor is well explained by a combination of others. It can inflate coefficient uncertainty and make individual coefficient estimates sensitive to sampling changes, even when predictions within the observed domain remain useful. Pairwise Pearson correlation alone can miss joint dependencies."),
         eq('VIFⱼ = 1 / (1 − Rⱼ²)'),
         p("For each original predictor, Rⱼ² is obtained from its auxiliary regression on the other original predictors and a constant. VIF is computed on the training partition, with an explicit intercept in the design matrix; the intercept's own VIF is omitted [6]. Thresholds such as 5 or 10 are heuristics and are not used as automatic feature-removal criteria."),
         table('Table 9. Original-input VIF on the training partition.',diag['vif'].round(3)),
         p("Predictors above VIF 5: " + ', '.join(diag['vif'].query('VIF > 5').Feature) + ". Ingredient masses participate in joint composition constraints, so interpreting a coefficient independently of the full mixture is particularly difficult."),
         p(f"Original-input VIF is not a diagnostic for every expanded polynomial column. The selected standardized expanded design, including intercept, has rank {stats['Expanded design rank']} of {stats['Expanded design columns']} columns and condition number {stats['Expanded condition number']:.2e}. This identifies substantial sensitivity despite full numerical rank. Standardization improves scale conditioning but cannot remove linear dependence."),
         p("All eight domain-relevant predictors remain in the required OLS models. Removing variables after inspecting the complete dataset would change the prespecified comparison and could omit useful interactions. Ridge regularization, grouped formulation validation and coefficient stability analysis are reasonable future experiments, with their own training-only tuning procedure."))
    wc = assoc.loc['Water_Cement_Ratio']
    page(h('16 Water Cement Ratio Analysis'),
         eq('Water-cement ratio = Water (kg/m³) / Cement (kg/m³)'),
         p(f"On the deduplicated observations, water-cement ratio has Pearson correlation {wc['Pearson r']:.4f} with strength (p = {wc['Pearson p']:.3g}) and Spearman correlation {wc['Spearman rho']:.4f} (p = {wc['Spearman p']:.3g}). These are unadjusted, exploratory associations. Repeated formulations and unmeasured conditions limit the independence assumptions behind the p-values."),
         fig('factor_trends','Figure 9. Observed ratio and age relationships; LOWESS is descriptive.',6.2),
         p("The ratio uses cement only in its denominator; it is not the water-to-total-cementitious-material ratio because slag and fly ash are excluded. This distinction matters when comparing mixtures containing supplementary materials. Excess water relative to cement can contribute to capillary porosity after hardening, while sufficient water and suitable curing are necessary for hydration [7,8]."),
         p("The observed negative association is consistent with the physical context but does not prove that changing water alone produces the observed strength difference. Cement amount, supplementary binders, age and admixtures can confound the association. The application's scenario tool changes water at fixed cement and holds other masses constant. Its curve is the selected model's conditional prediction, not a measured intervention or a certified mix-design recommendation."))
    age = assoc.loc['Age']
    age_table = enriched.groupby('Age Group',observed=True).Strength.agg(['count','mean','median','std']).round(3).reset_index()
    age_table.columns = ['Age group','Count','Mean MPa','Median MPa','SD MPa']
    page(h('17 Curing Age Analysis'),
         p(f"Curing age has Pearson correlation {age['Pearson r']:.4f} with strength and Spearman correlation {age['Spearman rho']:.4f}. The stronger rank relationship is consistent with an association that is not adequately summarized by a single linear slope. It does not establish the shape of a causal time-response curve."),
         table('Table 10. Observed strength summaries across curing periods.',age_table),
         fig('factor_distributions','Figure 10. Ratio distribution and observed age-group strength distributions.',6.2),
         p("Hydration can continue to develop strength as long as moisture, temperature and available reactants permit it [8]. The dataset records age rather than the actual curing regime. Comparing age groups therefore also compares potentially different mixtures. Within-group spread and unequal group sizes are important when interpreting group means."),
         p("Interactive scenarios begin with a real training mixture and vary age with all ingredient masses held fixed. The polynomial model is unconstrained and may show decreasing predictions at sparse late ages. Such behavior is a limitation of the estimated function, not a claim that well-cured concrete follows that trajectory. Scenarios should remain exploratory and should not be extrapolated to durability or structural safety outcomes."))
    improvement = (metrics.loc['Linear','Test RMSE']-selected['Test RMSE'])/metrics.loc['Linear','Test RMSE']*100
    page(h('18 Results and Discussion'),
         fig('actual_predicted','Figure 11. Selected model predictions on the unchanged test partition.',5.8),
         p(f"Under the recorded split, {meta['selected']} reduces test RMSE by {improvement:.1f}% relative to the linear baseline. Its training RMSE is {selected['Train RMSE']:.3f} MPa, test RMSE is {selected['Test RMSE']:.3f} MPa and CV standard deviation is {selected['CV SD']:.3f} MPa. The larger training-validation separation reflects variance risk as complexity increases. Degree 2 offers a smaller model and should remain a meaningful comparator, even when the minimum-CV rule selects degree 3."),
         p("The comparison supports the usefulness of nonlinear terms for this protocol. It does not show that all polynomial models generalize better, that degree 3 is statistically significantly superior, or that performance transfers to other concrete sources. The test R² indicates explained variation within this sample; it is not an engineering acceptance measure."),
         h('19 Conclusion'),
         p("The project demonstrates a complete SML workflow from original data to a reproducible, interactive prediction application. Polynomial features capture useful nonlinear structure, while residual departures, predictor dependence and sensitivity to validation folds limit certainty. Accurate reporting of these weaknesses is part of a successful regression study. SDG 9 motivates the material-analysis context, but deployment benefits require independent field validation."))
    page(h('20 Limitations'),
         p("The sample is modest relative to the 164 nonconstant degree 3 terms. The selected CV score is a model-selection estimate and can be optimistic because the degree is chosen using those folds. A single held-out split cannot quantify all sampling uncertainty; fold standard deviations are not confidence intervals."),
         p("Grouping protects identical input vectors, but the same formulation at different ages may cross partitions. No laboratory, production batch or site identifiers are available to establish complete independence. Unknown dependence can affect both generalization estimates and exploratory p-values."),
         p("Curing temperature, humidity, cement chemistry, test procedures and specimen geometry are not represented as model inputs. The data do not identify causal effects. Ingredient combinations within marginal training ranges can still be unlike any observed mixture. The model is not constrained to positive or monotone predictions and extrapolation can be unstable."),
         p("Normality and constant-variance assumptions are not asserted without evidence. The interface offers point estimates and aggregate error metrics, not calibrated individual prediction intervals. Joblib files must be treated as trusted local artifacts and regenerated if the dependency versions or source data change."),
         h('21 Future Scope'),
         p("Use formulation-level and source-level holdouts where identifiers are available; compare nested or repeated cross-validation; investigate ridge-regularized polynomial regression with training-only tuning; and examine sensitivity of the selected degree to the random split. These extensions should preserve a final untouched evaluation set."),
         p("Acquire controlled curing and material-provenance variables, evaluate external laboratory observations, and study scientifically motivated transformations or monotonic models. Calibrated uncertainty could be explored with conformal prediction under defensible exchangeability assumptions. Any proposed mix optimization would also need workability, durability, cost and engineering constraints."),
         p("Maintain the application as a reproducible teaching tool: retain source checksums, experiment metadata, split assignments, automated tests and matching report exports. Certified engineering use would require a substantially different validation and approval process."))
    page(h('22 References'),*[p(ref) for ref in REFERENCES],
         p("Web references accessed 1 November 2026. Numeric results in this report were generated by the included project code; no published model performance was substituted for execution results."),
         p(f"Experiment {meta['experiment_id']}; seed {meta['seed']}; {meta['folds']} folds. Dataset SHA-256: {meta['data_sha256']}."))
    return pages,diag
