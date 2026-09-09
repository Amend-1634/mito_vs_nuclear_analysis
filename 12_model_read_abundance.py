import pandas as pd
import numpy as np
import patsy
from itertools import combinations
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from scipy.stats import shapiro, chi2 as chi2_dist
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model comparison: for each of nuclear_to_mito_ratio and nuclear_to_plastid_ratio, model the
# raw nuclear read count directly with a log-link count distribution (Poisson vs Negative
# Binomial). Unlike a fixed-core comparison that always forces host-identity,
# log_after_filter_GTDB, and log_organellar into every formula and only varies {sample_age,
# feature, project_name}, this script lets every predictor recombine freely: host-identity
# (genus, genome_size_gb, or omitted entirely -- see section 2) and log_after_filter_GTDB are
# each independently optional, tried in every combination alongside every combination of
# {sample_age, feature, project_name}, and the best-supported model per response is selected by
# AIC across that entire combined search space. log_organellar (log of the organellar --
# mito/plastid -- read count) is always included rather than searched over (see section 3);
# its interaction with genus and with feature, when those are present, is offered as an
# independently optional term instead.

# ---------------------------------------------------------------------------
# 1. Load and build df_final_with_pattern
# ---------------------------------------------------------------------------
# read pivot file from 4_plot_nuclear_mito_material.py
pivot_table1 = pd.read_csv(
    '/Users/chenyjin/Documents/analysis/chap4_mito_nuclear/260513/true_genus_mito_plastid/filter_perc_id/pivot_table_n_reads.feature_material.csv',
    index_col=0).reset_index()

# read genome size (host-identity alternative to genus -- see section 2) and sum per genus
genome_size = pd.read_excel(
    '/Users/chenyjin/Documents/analysis/chap4_mito_nuclear/260513/true_genus_mito_plastid/filter_perc_id/species_genome_size.xlsx',
    header=0)
genome_size['genus'] = genome_size['species'].str.split().str[0]
genome_size_by_genus = genome_size.groupby('genus', as_index=False)['genome_size_gb'].sum()
pivot_table1 = pivot_table1.merge(genome_size_by_genus, on='genus', how='left')

# Combine Mammut/Mammuthus mito reads into Elephas, mirroring 5_model_comparison.py: ancient
# elephant mtDNA is often taxonomically classified as the extinct relatives Mammut/Mammuthus
# rather than Elephas itself, so their mito reads belong to the same underlying animal.
# pivot_table_n_reads.feature_material.csv (built by an earlier script) does not do this
# reassignment, which silently left Elephas's mito column at 0 for several samples (their true
# mito reads sat under Mammut/Mammuthus instead) -- undefined nuclear:mito ratio, so those rows
# were being dropped entirely. Zero out Mammut/Mammuthus's own mito afterward (rather than
# leaving their original value) to avoid double-counting the same reads under both labels. This
# must happen before the damage_authentication merge/filter below -- Mammut/Mammuthus rows for
# the affected samples don't themselves carry a 'with_pattern' label and would otherwise be
# dropped before their mito counts could be summed into Elephas.
mammut_mammuthus_mask = pivot_table1['genus'].isin(['Mammut', 'Mammuthus'])
mammut_mammuthus_mito_by_sample = pivot_table1.loc[mammut_mammuthus_mask].groupby('sample')['mito'].sum()
elephas_mask = pivot_table1['genus'] == 'Elephas'
pivot_table1.loc[elephas_mask, 'mito'] = (
    pivot_table1.loc[elephas_mask, 'mito'].fillna(0)
    + pivot_table1.loc[elephas_mask, 'sample'].map(mammut_mammuthus_mito_by_sample).fillna(0)
)
pivot_table1.loc[mammut_mammuthus_mask, 'mito'] = 0

# add damage information
# add damage plot authentication results /Users/chenyjin/Documents/analysis/chap4_mito_nuclear/260513/true_genus_mito_plastid/plot_damage_png/list_genus.damage.txt, delimited with space
df_damage = pd.read_csv(
    '/Users/chenyjin/Documents/analysis/chap4_mito_nuclear/260513/true_genus_mito_plastid/plot_damage_png/list_genus.damage.txt',
    header=None, sep="\t")
df_damage.columns = ['genus', 'taxid', 'sample', 'damage_authentication']

# Merge damage authentication results with pivot table
df_merge_damage = pivot_table1.merge(df_damage, on=['sample', 'genus'], how='left')

# add information of total number of reads after filtering
df_n_reads = pd.read_csv(
    '/Users/chenyjin/Documents/analysis/chap4_mito_nuclear/260513/true_genus_mito_plastid/filter_perc_id/TableS4_n_reads.csv',
    header=0)
df_n_reads = df_n_reads.dropna(subset=['sample'])

df_final = df_merge_damage.merge(df_n_reads[['sample', 'after_filtering_with_GTDB']], on='sample', how='left')

# filter for rows with damage_authentication as with_pattern
df_final_with_pattern = df_final[df_final['damage_authentication'] == 'with_pattern'].copy()

# ---------------------------------------------------------------------------
# 2. Define the three response variables and the fixed / candidate predictors. Mirroring
#    5_model_comparison.py, nuclear_to_mito_ratio is split into separate animal- and plant-
#    genus datasets rather than modeling both groups together with one genus effect: pooling
#    e.g. Capra/Elephas/Homo (each n=1) together with Populus/Salix (n=23/33) in a single
#    genus fixed effect was driving the singleton-genus separation/non-convergence issues
#    seen in the combined fits.
#
#    genus and genome_size_gb are treated as mutually exclusive alternatives for controlling
#    for host identity, never combined in the same formula: genome_size_gb is a per-genus
#    constant (summed across constituent species -- see the load step above), so it is exactly
#    collinear with C(genus) whenever both are available, the same reason feature/project_name
#    are kept mutually exclusive elsewhere in this analysis. genome_size_gb is only available
#    for the animal genera (species_genome_size.xlsx has no entries for the plant genera), so
#    it is only offered as a host-identity option where the data actually supports it. Section 3's
#    free-recombination search also allows host-identity to be omitted entirely (neither genus nor
#    genome_size_gb) -- it is a third option, not just a binary choice between the two.
# ---------------------------------------------------------------------------
ANIMAL_GENERA = {'Canis', 'Capra', 'Elephas', 'Gallus', 'Homo', 'Lepus',
                  'Mammut', 'Mammuthus', 'Martes', 'Phoxinus', 'Salmo', 'Sciurus'}
PLANT_GENERA = {'Populus', 'Salix', 'Betula'}

RESPONSE_CONFIGS = [
    ('animal_nuclear_mito', 'nuclear_to_mito_ratio', ANIMAL_GENERA),
    ('plant_nuclear_mito', 'nuclear_to_mito_ratio', PLANT_GENERA),
    ('plant_nuclear_plastid', 'nuclear_to_plastid_ratio', PLANT_GENERA),
]

DENOM_COL = {'nuclear_to_mito_ratio': 'mito', 'nuclear_to_plastid_ratio': 'plastid'}

FIXED_PREDICTORS = ['after_filtering_with_GTDB']  # kept in the dataset build so every candidate
                                                   # formula below is fit on identical rows --
                                                   # not forced into every formula (see FLEX_PREDICTORS)
HOST_PREDICTORS = ['genus', 'genome_size_gb']  # mutually exclusive host-identity control
FLEX_PREDICTORS = ['log_after_filter_GTDB']  # log_organellar used to live here too but is now
                                              # always included instead of searched over -- see
                                              # section 3
CANDIDATE_PREDICTORS = ['sample_age', 'feature', 'project_name']  # used for the dataset build's
                                                                   # NA-dropping and level counts,
                                                                   # and for section 4's GEE
                                                                   # clustering variable -- project_name
                                                                   # is kept here so its rows still
                                                                   # get the same NA-filtering even
                                                                   # though it's excluded from the
                                                                   # search itself below
SEARCH_CANDIDATE_PREDICTORS = ['sample_age', 'feature']  # mutually exclusive, like HOST_PREDICTORS
                                                          # -- at most one enters a formula at a
                                                          # time. project_name is excluded from
                                                          # section 3's search entirely (not offered
                                                          # as a choice at all), rather than merely
                                                          # kept mutually exclusive with feature
DIAGNOSTIC_ALPHA = 0.05
VIF_THRESHOLD = 10


def build_dataset(ratio_col, genus_set, require_genome_size):
    """Filter down to rows where nuclear and the corresponding organellar read count are usable
    (not both zero -- see the mixed-zero comment below), drop missing values, and restrict to the
    given genus_set (animal- vs. plant-genus datasets are modeled separately). All candidate
    predictors -- and genome_size_gb, when
    it's going to be offered as a host-identity alternative for this response -- are dropped for
    NA up front (not just the ones used in a given formula) so every model in the comparison for
    this response is fit on the identical set of rows, keeping AIC/BIC comparable across rows and
    across the genus/genome_size_gb host-identity choice."""
    denom_col = DENOM_COL[ratio_col]
    needed_cols = ['sample', 'genus', 'nuclear', denom_col] + FIXED_PREDICTORS + CANDIDATE_PREDICTORS
    if require_genome_size:
        needed_cols.append('genome_size_gb')
    sub = df_final_with_pattern.dropna(subset=needed_cols).copy()
    sub = sub[sub['genus'].isin(genus_set)]
    # Exclude only rows where BOTH nuclear and organellar reads are 0 (uninformative -- nothing
    # was recovered at all). Rows with nuclear==0 but organellar>0 (or vice versa) are kept: they
    # are informative structural zeros (e.g. organellar DNA detected in a sample too degraded for
    # any nuclear reads to map), not the kind of uninformative zero this filter is meant to drop.
    sub = sub[~((sub['nuclear'] == 0) & (sub[denom_col] == 0))]
    sub = sub[np.isfinite(sub['after_filtering_with_GTDB']) & (sub['after_filtering_with_GTDB'] > 0)]
    sub = sub[np.isfinite(sub['sample_age'])]
    sub['log_after_filter_GTDB'] = np.log10(sub['after_filtering_with_GTDB'])
    sub['sample_age_kyr'] = sub['sample_age'] / 1000.0
    sub['log_organellar'] = np.log(sub[denom_col] + 1)  # pseudocount of 1: denom_col == 0 is
                                                         # now a valid row (nuclear>0 side of the
                                                         # mixed-zero case above), and log(0) is
                                                         # undefined
    return sub


def term_for(p):
    if p in ('feature', 'project_name'):
        return f'C({p})'
    return p


def term_for_count(p):
    # sample_age is O(10^2-10^6) years while the other design columns are O(1-10), which
    # overflows the discrete NB model's Newton-Raphson optimizer (confirmed via
    # RuntimeWarning: overflow in exp / ConvergenceWarning for every sample_age-containing
    # formula on the raw scale). Fit on sample_age in thousands of years instead; this is a
    # pure rescaling, so it does not change the model's fit or interpretation beyond the
    # coefficient's units (per 1,000 years instead of per year).
    return 'sample_age_kyr' if p == 'sample_age' else term_for(p)


def host_term_for(host):
    return 'C(genus)' if host == 'genus' else 'genome_size_gb'


def all_subsets(items):
    for r in range(len(items) + 1):
        for combo in combinations(items, r):
            yield combo


def model_stats(fit, k):
    # AIC/BIC/AICc computed directly from log-likelihood and k = the actual number of estimated
    # parameters (len(fit.params)), rather than trusting each model class's own .aic/.bic: GLM's
    # own .bic is deviance-based by default (not comparable to the discrete models' likelihood-
    # based .bic), and neither GLM's nor the discrete NB's built-in .aic/.df_model-derived k
    # account for ZINB's extra zero-inflation parameter(s). Computing all three manually from
    # llf and an explicitly-passed k keeps every family (Poisson, Negative Binomial, Zero-
    # Inflated Negative Binomial) on the same footing. AICc (Burnham & Anderson) is the small-
    # sample-corrected AIC: plain AIC under-penalizes extra parameters when n/k is small (rule of
    # thumb: correct whenever n/k < 40), which is the regime every response here sits in (n=15-85
    # against formulas with up to a dozen C(genus) levels).
    n = fit.nobs
    aic = -2 * fit.llf + 2 * k
    bic = -2 * fit.llf + k * np.log(n)
    denom = n - k - 1
    aicc = aic + (2 * k * (k + 1)) / denom if denom > 0 else np.nan
    return aic, bic, aicc


def nb_lr_gof(y, mu, alpha, df_resid):
    """Likelihood-ratio chi-square goodness-of-fit test of the Negative-Binomial (NB2) variance
    function, replacing a Pearson chi2/df_resid statistic (dispersion_ratio) with the deviance --
    2*(saturated_llf - model_llf) -- which for NB2 is exactly statsmodels' GLM NegativeBinomial
    family deviance, tested against chi2(df_resid). Used unmodified for ZINB rows too: each
    family's saturated per-observation log-likelihood term at y=0 is exactly 0 (NB2: the gammaln
    terms cancel and log(1/(1+alpha*0))=0; ZINB: the structural-zero probability that maximizes
    P(Y=0) is 1, i.e. log(1)=0), so the two families' deviance formulas coincide term-by-term.
    """
    if df_resid <= 0:
        return np.nan, np.nan
    lr_stat = sm.families.NegativeBinomial(alpha=alpha).deviance(np.asarray(y), np.asarray(mu))
    return lr_stat, chi2_dist.sf(lr_stat, df_resid)


def poisson_pearson_gof(resid_pearson, df_resid):
    """Pearson chi-square goodness-of-fit test of the Poisson variance function: sum of squared
    Pearson residuals tested against chi2(df_resid). Poisson has no dispersion parameter (var ==
    mean is fixed, not estimated), so there is no NB2-style deviance formula for it to plug into
    nb_lr_gof -- the classic Pearson statistic is the standard GOF check for a plain Poisson GLM,
    kept as a separate test rather than forced into the deviance/LR framework used for NB/ZINB.
    """
    if df_resid <= 0:
        return np.nan, np.nan
    stat = np.sum(np.asarray(resid_pearson) ** 2)
    return stat, chi2_dist.sf(stat, df_resid)


def fit_zinb(formula, sub, count_start, alpha_start, exog_infl_formula='1', inflate_start=None):
    # Zero-Inflated Negative Binomial via ZeroInflatedNegativeBinomialP: models the response as a
    # mixture of an always-zero ("structural zero") process and an ordinary Negative Binomial
    # count process, rather than assuming every zero is an ordinary low NB draw. exog_infl_formula
    # defaults to an intercept-only inflation model (a single constant probability of a structural
    # zero shared by every row) -- the minimal test of whether zero-inflation improves the fit at
    # all, not a full characterization of which rows are more likely to be structural zeros.
    # Plain BFGS from a naive start reliably fails to converge on this likelihood (confirmed
    # empirically), so this warm-starts from a Nelder-Mead pass (robust to a poor starting point,
    # but slow/imprecise near the optimum) initialized from caller-supplied count/alpha/inflation
    # starting values (e.g. an already-fit plain-NB's coefficients for the main search, or a
    # hand-picked guess for the smaller auxiliary link-test regression, whose formula/exog shape
    # differs from whatever formula supplied count_start), then refines with BFGS from the NM
    # result.
    endog_z, exog_z = patsy.dmatrices(formula, sub, return_type='dataframe')
    exog_infl_z = patsy.dmatrix(exog_infl_formula, sub, return_type='dataframe')
    model = ZeroInflatedNegativeBinomialP(endog_z, exog_z, exog_infl=exog_infl_z, inflation='logit', p=2)
    n_infl = exog_infl_z.shape[1]
    if inflate_start is None:
        inflate_start = np.concatenate([[np.log(0.05 / 0.95)], np.zeros(n_infl - 1)])
    else:
        inflate_start = np.atleast_1d(inflate_start)
    start = np.concatenate([inflate_start, count_start, [alpha_start]])
    fit_nm = model.fit(start_params=start, method='nm', maxiter=3000, maxfun=6000, disp=0)
    fit = model.fit(start_params=fit_nm.params, method='bfgs', maxiter=1000, disp=0)
    if not np.isfinite(fit.llf):
        raise ValueError('ZINB fit produced a non-finite log-likelihood')
    return fit, exog_z


# ---------------------------------------------------------------------------
# 3. Count-based GLM comparison: model the raw numerator read count (nuclear) directly with a
#    log-link count distribution, with log(organellar read count) -- mito or plastid,
#    respectively -- entered as a *free* covariate (log_organellar, always included -- see
#    below) rather than fixed as an offset. An offset assumes proportionality (coefficient
#    forced to 1, i.e. nuclear reads scale exactly with organellar reads); putting it in as a
#    free predictor instead *tests* that assumption -- the fitted coefficient's deviation from 1
#    (Wald test below) tells you whether the nuclear:organellar relationship really is
#    proportional (isometric) or some other allometric scaling. Poisson assumes variance == mean;
#    Negative Binomial (MLE-estimated
#    dispersion via the discrete model, so alpha is fit rather than fixed) relaxes that, so
#    comparing the two per formula is a direct overdispersion check. Zero-Inflated Negative
#    Binomial (ZINB, intercept-only inflation -- see fit_zinb() above) is fit as a third family
#    alongside Poisson and NB per formula, whenever the plain NB fit succeeds (it supplies the
#    ZINB warm-start): a diagnostic check of this whole comparison found the observed zero count
#    in nuclear substantially exceeds what NB's own variance function predicts for two of the
#    three responses, so zero-inflation is tested formula-by-formula here rather than assumed
#    away or bolted on only after the fact.
#
#    Every predictor recombines freely here, rather than a fixed core plus a varying remainder:
#    host-identity (genus, genome_size_gb, or omitted -- host_choices) and log_after_filter_GTDB
#    (FLEX_PREDICTORS) are each independently included or dropped, crossed with a single
#    candidate choice (sample_age, feature, or none -- candidate_choices, mutually exclusive for
#    the same collinearity reason as host). project_name is excluded from this search entirely
#    (see SEARCH_CANDIDATE_PREDICTORS) rather than offered as a fourth mutually-exclusive option;
#    it is still used in section 4's GEE as the clustering variable. host and candidate each
#    contribute at most one term at a time, but "none" is a valid option for both.
#
#    log_organellar (log of the organellar -- mito/plastid -- read count) is always included,
#    unlike the rest of FLEX_PREDICTORS, rather than searched over: every formula tests the
#    nuclear:organellar scaling relationship (see the Wald test below). Its interaction with
#    genus and with feature is each independently optional, offered only when that term is
#    itself present in the formula (host == 'genus' / candidate == 'feature' respectively) --
#    an interaction with a term that isn't in the model wouldn't be identifiable. So the search
#    space is len(host_choices) x 2**len(FLEX_PREDICTORS) x len(candidate_choices) x (2 if
#    host == 'genus' else 1) x (2 if candidate == 'feature' else 1) formulas per response.
# ---------------------------------------------------------------------------
count_rows = []
count_fits = {}  # (response_name, family_name, formula) -> fitted result

for response_name, ratio_col, genus_set in RESPONSE_CONFIGS:
    genome_size_available = df_final_with_pattern.loc[
        df_final_with_pattern['genus'].isin(genus_set), 'genome_size_gb'].notna().any()
    host_options = ['genus', 'genome_size_gb'] if genome_size_available else ['genus']
    host_choices = [None] + host_options  # None = host-identity term omitted entirely
    candidate_choices = [None] + SEARCH_CANDIDATE_PREDICTORS  # None = no candidate term at all

    sub = build_dataset(ratio_col, genus_set, require_genome_size=genome_size_available)

    n_levels = {p: sub[p].nunique() for p in CANDIDATE_PREDICTORS}
    n_genus_levels = sub['genus'].nunique()

    for host in host_choices:
        if host == 'genus' and n_genus_levels < 2:
            continue  # can't estimate a genus effect with <2 levels present

        host_terms = [host_term_for(host)] if host is not None else []
        # log_organellar:C(genus) is only identifiable when C(genus) is itself in the formula
        genus_interaction_choices = [False, True] if host == 'genus' else [False]

        for flex_subset in all_subsets(FLEX_PREDICTORS):
            for candidate in candidate_choices:
                if candidate == 'feature' and n_levels[candidate] < 2:
                    continue

                candidate_terms = [term_for_count(candidate)] if candidate is not None else []
                # log_organellar:C(feature) is only identifiable when C(feature) is itself in
                # the formula
                feature_interaction_choices = [False, True] if candidate == 'feature' else [False]

                for include_genus_interaction in genus_interaction_choices:
                  for include_feature_interaction in feature_interaction_choices:
                    interaction_terms = []
                    if include_genus_interaction:
                        interaction_terms.append('log_organellar:C(genus)')
                    if include_feature_interaction:
                        interaction_terms.append('log_organellar:C(feature)')

                    # log_organellar is always included (see section 3's header comment), so
                    # terms is never empty and the 'nuclear ~ 1' fallback previously needed here
                    # is gone
                    terms = host_terms + list(flex_subset) + ['log_organellar'] + candidate_terms + interaction_terms
                    formula = 'nuclear ~ ' + ' + '.join(terms)

                    try:
                        fit_pois = smf.glm(formula, data=sub, family=sm.families.Poisson()).fit()
                    except Exception:
                        continue
                    try:
                        fit_nb = smf.negativebinomial(formula, data=sub).fit(disp=0, maxiter=200)
                        if not (fit_nb.mle_retvals.get('converged', True) and np.isfinite(fit_nb.llf)):
                            fit_nb = None
                    except Exception:
                        fit_nb = None

                    zinb_result = None  # (fit, exog) or None -- ZINB needs fit_nb's coefficients
                                         # as a warm start, so it's skipped whenever NB itself
                                         # failed
                    if fit_nb is not None:
                        try:
                            zinb_result = fit_zinb(formula, sub, fit_nb.params.values[:-1], fit_nb.params['alpha'])
                        except Exception:
                            zinb_result = None

                    # Likelihood-ratio test of Poisson (alpha=0, a boundary value) vs Negative
                    # Binomial. Since the null sits on the boundary of the alpha parameter space,
                    # the usual chi2(1) p-value is halved (Cameron & Trivedi's standard boundary
                    # correction).
                    lr_stat = lr_pvalue = alpha_nb = np.nan
                    if fit_nb is not None:
                        lr_stat = max(2 * (fit_nb.llf - fit_pois.llf), 0.0)
                        lr_pvalue = 0.5 * chi2_dist.sf(lr_stat, 1)
                        alpha_nb = fit_nb.params.get('alpha', np.nan)

                    fits_to_record = [('poisson', fit_pois)]
                    if fit_nb is not None:
                        fits_to_record.append(('negativebinomial', fit_nb))
                    if zinb_result is not None:
                        fits_to_record.append(('zeroinflatednegativebinomial', zinb_result[0]))

                    for family_name, fit in fits_to_record:
                        is_zinb = family_name == 'zeroinflatednegativebinomial'

                        # Wald test of H0: log_organellar's coefficient == 1, i.e. an exactly
                        # proportional (isometric) nuclear:organellar relationship. Rejecting H0
                        # means the "ratio" is not actually scale-invariant in this data --
                        # nuclear reads scale faster or slower than organellar reads rather than
                        # in fixed proportion. log_organellar is always in the formula now, so
                        # this should always succeed; the try/except is kept defensively.
                        try:
                            wald = fit.t_test('log_organellar = 1')
                            wald_stat = float(np.ravel(wald.tvalue)[0])
                            wald_pvalue = float(np.ravel(wald.pvalue)[0])
                        except Exception:
                            wald_stat = wald_pvalue = np.nan

                        # Regression-assumption diagnostics for all three count-model families
                        # (Poisson, Negative Binomial, Zero-Inflated Negative Binomial), mirroring
                        # the OLS diagnostics in 5_model_comparison.py but adapted to a GLM. Checks
                        # homoscedasticity (Breusch-Pagan, run on deviance residuals rather than
                        # Pearson residuals -- deviance residuals are the standard choice for BP on
                        # a GLM/count model since they're the ones the model's own fitting/deviance
                        # machinery is built around, and they're less dominated by the handful of
                        # extreme high-count rows that dominate Pearson residuals for data spanning
                        # this many orders of magnitude) and multicollinearity (max VIF).
                        # Normality (Shapiro-Wilk, still on Pearson residuals) is reported for
                        # reference but deliberately excluded from assumptions_ok: none of these
                        # families has "residuals are normal" as an actual assumption the way
                        # classical OLS does. dispersion_ratio (Pearson chi2/df_resid) is the
                        # complementary, family-specific check of whether that family's own variance
                        # function is well-calibrated (should be close to 1; Poisson's is expected to
                        # fail given the overdispersion already established, which is exactly why NB
                        # and ZINB were tried as alternatives). ZINB has no built-in
                        # .resid_pearson/.model.exog the way Poisson/NB do (its mean/variance
                        # functions blend the count and zero-inflation components), so its residuals
                        # and design matrix are pulled from predict(which=...) and the exog captured
                        # at fit time instead; its deviance residuals reuse the same NB2 formula as
                        # nb_lr_gof (see that function's docstring for why this is valid) evaluated
                        # at the marginal (zero-inflation-adjusted) mean.
                        if is_zinb:
                            zinb_exog = zinb_result[1]
                            mu_hat = fit.predict(which='mean')
                            var_hat = fit.predict(which='var')
                            resid = (sub['nuclear'].values - mu_hat) / np.sqrt(var_hat)
                            resid_dev = sm.families.NegativeBinomial(alpha=fit.params['alpha']).resid_dev(
                                sub['nuclear'].values, mu_hat)
                            exog_arr = zinb_exog.values
                            exog_names = list(zinb_exog.columns)
                            df_resid = fit.df_resid
                        elif family_name == 'poisson':
                            resid = fit.resid_pearson
                            resid_dev = fit.resid_deviance
                            exog_arr = fit.model.exog
                            exog_names = fit.model.exog_names
                            df_resid = fit.df_resid
                        else:  # negativebinomial
                            resid = fit.resid_pearson
                            # fit.predict(which='mean'), not fit.fittedvalues -- see the nb_lr_gof
                            # call site's comment above for why fittedvalues is wrong here.
                            resid_dev = sm.families.NegativeBinomial(alpha=fit.params['alpha']).resid_dev(
                                sub['nuclear'].values, fit.predict(which='mean'))
                            exog_arr = fit.model.exog
                            exog_names = fit.model.exog_names
                            df_resid = fit.df_resid
                        try:
                            _, p_homoscedasticity, _, _ = het_breuschpagan(resid_dev, exog_arr)
                        except Exception:
                            p_homoscedasticity = np.nan
                        _, p_normality = shapiro(resid) if len(resid) >= 3 else (np.nan, np.nan)
                        dispersion_ratio = np.sum(resid ** 2) / df_resid if df_resid > 0 else np.nan
                        try:
                            # exog_names can have extra trailing entries beyond what's actually in
                            # exog_arr's columns (e.g. the discrete NB model appends 'alpha', the
                            # dispersion parameter, to exog_names but not to exog) -- index by
                            # exog_arr's own column count, not by enumerating exog_names.
                            with np.errstate(divide='ignore', invalid='ignore'):
                                vifs = [variance_inflation_factor(exog_arr, i)
                                        for i in range(exog_arr.shape[1]) if exog_names[i] != 'Intercept']
                            max_vif = max(vifs) if vifs else np.nan
                        except Exception:
                            max_vif = np.nan
                        assumptions_ok = (
                            (np.isnan(p_homoscedasticity) or p_homoscedasticity > DIAGNOSTIC_ALPHA)
                            and (np.isnan(max_vif) or max_vif < VIF_THRESHOLD)
                        )

                        # Direct tests of each family's own claimed assumptions, for every row (not
                        # just the eventual "best" one): NB/ZINB get both a mean-function test (log
                        # link, correct predictors, via a link test) and a variance-function test
                        # (likelihood-ratio chi-square goodness-of-fit -- nb_lr_gof, deviance vs
                        # chi2(df_resid)). Poisson has a fixed variance function (var == mean, no
                        # dispersion parameter to test via deviance the same way), so it gets only
                        # the classic Pearson chi-square goodness-of-fit test instead
                        # (poisson_pearson_gof); it has no mean_function_ok / link test, same as
                        # before. mle_retvals's own 'converged' flag is unreliable for the small
                        # auxiliary link-test fit (BFGS often reports warnflag=2 "precision loss" on
                        # fits that are otherwise fine), so usability is judged by finiteness of the
                        # log-likelihood/p-values instead.
                        linktest_eta_coef = linktest_eta_sq_coef = linktest_eta_sq_pvalue = np.nan
                        mean_function_ok = lr_stat_gof = gof_pvalue = variance_function_ok = np.nan
                        if family_name == 'negativebinomial':
                            eta_hat = fit.model.exog @ fit.params[:-1]  # drop the trailing 'alpha' param
                            link_test_df = pd.DataFrame({'nuclear': sub['nuclear'].values, 'eta_hat': eta_hat,
                                                          'eta_hat_sq': eta_hat ** 2})
                            try:
                                link_fit = smf.negativebinomial('nuclear ~ eta_hat + eta_hat_sq', data=link_test_df).fit(
                                    disp=0, maxiter=500, method='bfgs')
                                linktest_eta_coef = link_fit.params['eta_hat']
                                linktest_eta_sq_coef = link_fit.params['eta_hat_sq']
                                linktest_eta_sq_pvalue = link_fit.pvalues['eta_hat_sq']
                                if not (np.isfinite(link_fit.llf) and np.isfinite(linktest_eta_sq_pvalue)):
                                    raise ValueError('link test produced non-finite results')
                                mean_function_ok = linktest_eta_sq_pvalue > DIAGNOSTIC_ALPHA
                            except Exception:
                                linktest_eta_coef = linktest_eta_sq_coef = linktest_eta_sq_pvalue = np.nan
                                mean_function_ok = np.nan

                            # fit.fittedvalues for statsmodels' discrete NegativeBinomial is
                            # documented as the linear predictor Xb (log scale), NOT the
                            # response-scale mean -- unlike GLM's fittedvalues. nb_lr_gof needs mu
                            # on the response scale, so use predict(which='mean') (exp(Xb))
                            # instead, matching what ZINB already does below via predict(which=...).
                            lr_stat_gof, gof_pvalue = nb_lr_gof(sub['nuclear'].values,
                                                                 fit.predict(which='mean'),
                                                                 fit.params['alpha'], df_resid)
                            variance_function_ok = gof_pvalue > DIAGNOSTIC_ALPHA if not np.isnan(gof_pvalue) else np.nan
                        elif is_zinb:
                            eta_hat = fit.predict(which='linear')
                            link_test_df = pd.DataFrame({'nuclear': sub['nuclear'].values, 'eta_hat': eta_hat,
                                                          'eta_hat_sq': eta_hat ** 2})
                            try:
                                # count_start = [intercept=0, eta_hat coef=1, eta_hat_sq coef=0]: if
                                # the outer ZINB's mean function is correctly specified, eta_hat alone
                                # should already reproduce it almost exactly, making this a good
                                # starting guess regardless of how complex the outer formula was.
                                link_fit, _ = fit_zinb('nuclear ~ eta_hat + eta_hat_sq', link_test_df,
                                                        np.array([0.0, 1.0, 0.0]), fit.params['alpha'],
                                                        inflate_start=fit.params['inflate_Intercept'])
                                linktest_eta_coef = link_fit.params['eta_hat']
                                linktest_eta_sq_coef = link_fit.params['eta_hat_sq']
                                linktest_eta_sq_pvalue = link_fit.pvalues['eta_hat_sq']
                                if not np.isfinite(linktest_eta_sq_pvalue):
                                    raise ValueError('link test produced non-finite p-value')
                                mean_function_ok = linktest_eta_sq_pvalue > DIAGNOSTIC_ALPHA
                            except Exception:
                                linktest_eta_coef = linktest_eta_sq_coef = linktest_eta_sq_pvalue = np.nan
                                mean_function_ok = np.nan

                            lr_stat_gof, gof_pvalue = nb_lr_gof(sub['nuclear'].values, mu_hat,
                                                                 fit.params['alpha'], df_resid)
                            variance_function_ok = gof_pvalue > DIAGNOSTIC_ALPHA if not np.isnan(gof_pvalue) else np.nan
                        elif family_name == 'poisson':
                            # lr_stat_gof/gof_pvalue hold the Pearson chi-square statistic/p-value
                            # here, not a deviance/LR statistic -- see poisson_pearson_gof().
                            lr_stat_gof, gof_pvalue = poisson_pearson_gof(resid, df_resid)
                            variance_function_ok = gof_pvalue > DIAGNOSTIC_ALPHA if not np.isnan(gof_pvalue) else np.nan

                        k = len(fit.params)
                        aic_val, bic_val, aicc_val = model_stats(fit, k)
                        row = {
                            'response': response_name,
                            'family': family_name,
                            'host_predictor': host if host is not None else 'none',
                            'formula': formula,
                            'includes_gtdb': 'log_after_filter_GTDB' in flex_subset,
                            'includes_age': candidate == 'sample_age',
                            'includes_feature': candidate == 'feature',
                            'includes_project_name': candidate == 'project_name',
                            'includes_genus_interaction': include_genus_interaction,
                            'includes_feature_interaction': include_feature_interaction,
                            'n_obs': int(fit.nobs),
                            'k': k,
                            'df_model': getattr(fit, 'df_model', np.nan),
                            'aic': aic_val,
                            'bic': bic_val,
                            'aicc': aicc_val,
                            'log_likelihood': fit.llf,
                            'deviance': getattr(fit, 'deviance', np.nan),
                            'pearson_chi2': getattr(fit, 'pearson_chi2', np.nan),
                            'dispersion_ratio': dispersion_ratio,
                            'alpha_nb': fit.params.get('alpha', np.nan) if family_name in
                                ('negativebinomial', 'zeroinflatednegativebinomial') else np.nan,
                            'zi_inflate_intercept': fit.params.get('inflate_Intercept', np.nan) if is_zinb else np.nan,
                            'lr_stat_nb_vs_poisson': lr_stat,
                            'lr_pvalue_nb_vs_poisson': lr_pvalue,
                            'wald_stat_log_organellar_eq1': wald_stat,
                            'wald_pvalue_log_organellar_eq1': wald_pvalue,
                            'p_homoscedasticity': p_homoscedasticity,
                            'p_normality': p_normality,
                            'max_vif': max_vif,
                            'assumptions_ok': assumptions_ok,
                            'linktest_eta_coef': linktest_eta_coef,
                            'linktest_eta_sq_coef': linktest_eta_sq_coef,
                            'linktest_eta_sq_pvalue': linktest_eta_sq_pvalue,
                            'mean_function_ok': mean_function_ok,
                            'lr_stat_gof': lr_stat_gof,
                            'gof_pvalue': gof_pvalue,
                            'variance_function_ok': variance_function_ok,
                        }
                        for term in fit.params.index:
                            row[f'coef_{term}'] = fit.params[term]
                            row[f'se_{term}'] = fit.bse[term]
                            row[f'pval_{term}'] = fit.pvalues[term]
                        count_rows.append(row)
                        count_fits[(response_name, family_name, formula)] = fit

count_comparison = pd.DataFrame(count_rows)
# Model selection below uses AICc (small-sample-corrected AIC), not raw AIC -- see aicc()'s
# docstring-comment above for why. delta_aic/is_best_aic are kept as delta_aicc/is_best_aicc to
# make clear which criterion is actually driving selection; raw aic/bic columns are still
# exported for reference.
count_comparison['delta_aicc'] = count_comparison.groupby(['response', 'family'])['aicc'].transform(lambda x: x - x.min())
count_comparison['is_best_aicc'] = (
    count_comparison.groupby(['response', 'family'])['aicc'].transform('min') == count_comparison['aicc'])
count_comparison.drop(
    columns=['host_predictor', 'includes_gtdb', 'includes_age', 'includes_feature',
             'includes_project_name', 'includes_genus_interaction', 'includes_feature_interaction']
).sort_values(['aicc', 'response']).to_csv(
    '/Users/chenyjin/Documents/analysis/chap4_mito_nuclear/260513/true_genus_mito_plastid/filter_perc_id/plots/new/count_predictor_model_comparison.include_0.csv',
    index=False)

print("\n=== Poisson vs Negative-Binomial count-GLM comparison (log_organellar = log(organellar "
      "read count), fit as a free covariate to test proportionality) ===")
print(count_comparison[['response', 'family', 'host_predictor', 'formula', 'n_obs', 'aic', 'aicc', 'bic',
                         'dispersion_ratio', 'coef_log_organellar', 'wald_pvalue_log_organellar_eq1',
                         'lr_stat_nb_vs_poisson', 'lr_pvalue_nb_vs_poisson', 'is_best_aicc']]
      .sort_values(['response', 'family', 'aicc']).to_string(index=False))

print("\n=== Count-model (Poisson / Negative-Binomial) regression-assumption diagnostics ===")
print(count_comparison
      [['response', 'family', 'host_predictor', 'formula', 'n_obs', 'p_homoscedasticity', 'p_normality',
        'max_vif', 'dispersion_ratio', 'assumptions_ok']]
      .sort_values(['response', 'family', 'assumptions_ok'], ascending=[True, True, False]).to_string(index=False))

# ---------------------------------------------------------------------------
# 4. GEE population-averaged count model, clustering on project_name only (not feature -- the
#    two are almost perfectly confounded, each project mapping to essentially one feature type,
#    so only one of them is used for grouping, and it is excluded from the fixed part below to
#    avoid reintroducing that confound). This is not a true random-effects model: GEE estimates
#    population-averaged fixed-effect coefficients and corrects their standard errors for
#    within-project correlation via a robust "sandwich" variance under an assumed exchangeable
#    working correlation -- it does not produce an actual variance component/ICC for
#    project_name the way a real mixed model would. A true random-intercept Poisson/Negative-
#    Binomial model was attempted first (PoissonBayesMixedGLM) but failed to converge for every
#    response tested here (implausible, wildly inconsistent variance-component estimates even
#    after standardizing the design matrix), confirming that tool's known immaturity for this
#    kind of data; GEE is the practical, reliably-converging alternative instead.
# ---------------------------------------------------------------------------
GEE_FIXED_FORMULA = 'nuclear ~ C(genus) + log_after_filter_GTDB + log_organellar + sample_age_kyr'

gee_rows = []
gee_fits = {}  # (response_name, family_name) -> fitted GEEResults

for response_name, ratio_col, genus_set in RESPONSE_CONFIGS:
    sub = build_dataset(ratio_col, genus_set, require_genome_size=False)

    # GEE's NegativeBinomial family takes alpha as a fixed input rather than estimating it
    # jointly (unlike the discrete MLE model in section 3), so borrow the MLE alpha from the
    # same fixed formula fit without any grouping structure as a plug-in dispersion estimate.
    try:
        alpha_fit = smf.negativebinomial(GEE_FIXED_FORMULA, data=sub).fit(disp=0, maxiter=200)
        alpha_hat = alpha_fit.params['alpha'] if np.isfinite(alpha_fit.llf) else 1.0
    except Exception:
        alpha_hat = 1.0

    group_sizes = sub['project_name'].value_counts()
    print(f"\n=== GEE comparison for {response_name} (n={len(sub)}, fixed part: {GEE_FIXED_FORMULA}, "
          f"{sub['project_name'].nunique()} project_name groups, plug-in NB alpha={alpha_hat:.4f}) ===")

    for family_name, family in [('poisson', sm.families.Poisson()),
                                 ('negativebinomial', sm.families.NegativeBinomial(alpha=alpha_hat))]:
        try:
            fit = smf.gee(GEE_FIXED_FORMULA, groups='project_name', data=sub,
                          family=family, cov_struct=sm.cov_struct.Exchangeable()).fit(scale='X2')
        except Exception as e:
            print(f"  family={family_name}: FAILED to fit ({e})")
            continue

        qic, qicu = fit.qic(scale=fit.scale)
        gee_fits[(response_name, family_name)] = fit
        row = {
            'response': response_name,
            'family': family_name,
            'fixed_formula': GEE_FIXED_FORMULA,
            'alpha_nb': alpha_hat if family_name == 'negativebinomial' else np.nan,
            'n_groups': sub['project_name'].nunique(),
            'min_group_size': int(group_sizes.min()),
            'max_group_size': int(group_sizes.max()),
            'n_obs': int(fit.nobs),
            'scale': fit.scale,
            'qic': qic,
            'qicu': qicu,
        }
        for term in fit.params.index:
            row[f'coef_{term}'] = fit.params[term]
            row[f'se_{term}'] = fit.bse[term]
            row[f'pval_{term}'] = fit.pvalues[term]
        gee_rows.append(row)

        print(f"\n  --- family={family_name} ({sub['project_name'].nunique()} groups, "
              f"sizes {group_sizes.min()}-{group_sizes.max()}) ---")
        print(f"  QIC={qic:.1f}  QICu={qicu:.1f}  scale={fit.scale:.3f}")
        print(fit.summary().tables[1])

gee_comparison = pd.DataFrame(gee_rows)
# qicu (the simplified, independence-working-correlation QIC variant), not qic, is used for
# comparison here: qic's trace-correction term is numerically degenerate for these fits (values
# up to ~1e15, versus a well-behaved qicu in the low thousands) -- see the GOF-investigation
# discussion for this script. qic/delta_qic are still exported to the CSV for reference but
# should not be trusted for model comparison.
gee_comparison['delta_qicu'] = gee_comparison.groupby('family')['qicu'].transform(lambda x: x - x.min())
gee_comparison['delta_qic'] = gee_comparison.groupby('family')['qic'].transform(lambda x: x - x.min())
gee_comparison.to_csv(
    '/Users/chenyjin/Documents/analysis/chap4_mito_nuclear/260513/true_genus_mito_plastid/filter_perc_id/plots/new/count_gee_comparison.include_0.csv',
    index=False)

print("\n=== GEE summary (project_name clustering, Poisson vs Negative-Binomial) ===")
print(gee_comparison[['response', 'family', 'n_groups', 'min_group_size', 'max_group_size',
                       'qicu', 'delta_qicu', 'scale']].to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Select the single best-supported model per response (lowest AICc across Poisson, Negative
#    Binomial, and Zero-Inflated Negative Binomial, and across the genus/genome_size_gb host-
#    identity choice). AICc rather than raw AIC because every response here has n/k well under
#    the usual n/k<40 rule-of-thumb threshold for the correction to matter (see model_stats()
#    above) -- the raw-AIC search space includes formulas with many-level C(genus)/C(feature)
#    terms against as few as 15 rows. Section 3's search never generates a formula combining
#    feature and project_name (near-perfectly confounded -- see CANDIDATE_PREDICTORS), so no
#    post-hoc exclusion is needed here. All three families already live on the same response
#    scale (the raw nuclear count) with the same log link, so their AICc is directly comparable
#    with no further correction needed.
# ---------------------------------------------------------------------------
dist_rows = []
reported_best_keys = []  # (response, family, formula) for the row actually reported as "the"
                          # best model per response -- used to flag is_reported_best_model in
                          # the merged table (section 6) below.

for response_name, ratio_col, genus_set in RESPONSE_CONFIGS:
    candidates = count_comparison[count_comparison['response'] == response_name]
    if len(candidates) == 0:
        print(f"\nNote: no converged model for {response_name} -- excluded from this comparison.")
        continue

    best = candidates.loc[candidates['aicc'].idxmin()]
    reported_best_keys.append((response_name, best['family'], best['formula']))

    pois_candidates = candidates[candidates['family'] == 'poisson']
    nb_candidates = candidates[candidates['family'] == 'negativebinomial']
    zinb_candidates = candidates[candidates['family'] == 'zeroinflatednegativebinomial']
    pois_best = pois_candidates.loc[pois_candidates['aicc'].idxmin()] if len(pois_candidates) else None
    nb_best = nb_candidates.loc[nb_candidates['aicc'].idxmin()] if len(nb_candidates) else None
    zinb_best = zinb_candidates.loc[zinb_candidates['aicc'].idxmin()] if len(zinb_candidates) else None

    dist_rows.append({
        'response': response_name,
        'poisson_formula': pois_best['formula'] if pois_best is not None else None,
        'poisson_host_predictor': pois_best['host_predictor'] if pois_best is not None else None,
        'poisson_aic': pois_best['aic'] if pois_best is not None else np.nan,
        'poisson_aicc': pois_best['aicc'] if pois_best is not None else np.nan,
        'poisson_bic': pois_best['bic'] if pois_best is not None else np.nan,
        'nb_formula': nb_best['formula'] if nb_best is not None else None,
        'nb_host_predictor': nb_best['host_predictor'] if nb_best is not None else None,
        'nb_aic': nb_best['aic'] if nb_best is not None else np.nan,
        'nb_aicc': nb_best['aicc'] if nb_best is not None else np.nan,
        'nb_bic': nb_best['bic'] if nb_best is not None else np.nan,
        'zinb_formula': zinb_best['formula'] if zinb_best is not None else None,
        'zinb_host_predictor': zinb_best['host_predictor'] if zinb_best is not None else None,
        'zinb_aic': zinb_best['aic'] if zinb_best is not None else np.nan,
        'zinb_aicc': zinb_best['aicc'] if zinb_best is not None else np.nan,
        'zinb_bic': zinb_best['bic'] if zinb_best is not None else np.nan,
        'zinb_p_structural_zero': (1 / (1 + np.exp(-zinb_best['zi_inflate_intercept'])))
            if zinb_best is not None else np.nan,
        'preferred_family': best['family'],
        'preferred_host_predictor': best['host_predictor'],
        'preferred_formula': best['formula'],
    })

distribution_comparison = pd.DataFrame(dist_rows)
distribution_comparison.to_csv(
    '/Users/chenyjin/Documents/analysis/chap4_mito_nuclear/260513/true_genus_mito_plastid/filter_perc_id/plots/new/count_poisson_vs_nb_comparison.include_0.csv',
    index=False)

print("\n=== Poisson vs Negative-Binomial vs ZINB AICc comparison (best model per family per response) ===")
print(distribution_comparison[['response', 'poisson_aicc', 'nb_aicc', 'zinb_aicc', 'preferred_family',
                                'preferred_host_predictor']].to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Single merged summary-statistics table across every fitted count model (Poisson, Negative
#    Binomial, and ZINB, every host-identity choice), sorted by response and then ascending AICc
#    within each response so the best-supported row per response sits at the top of its group.
#    The includes_*/is_reported_best_model bookkeeping columns used to build/order this table are
#    dropped from the export itself -- they're derivable from formula/family and just add clutter.
# ---------------------------------------------------------------------------
all_distributions_comparison = count_comparison.copy()
is_reported_best_model = all_distributions_comparison.apply(
    lambda r: (r['response'], r['family'], r['formula']) in reported_best_keys, axis=1)
n_flagged = int(is_reported_best_model.sum())

# Drop the includes_*/host_predictor bookkeeping columns from this export -- they're derivable
# from formula/family and just add clutter to the full per-model table.
all_distributions_comparison = all_distributions_comparison.drop(
    columns=['includes_gtdb', 'includes_age', 'includes_feature', 'includes_project_name',
             'includes_genus_interaction', 'includes_feature_interaction', 'host_predictor'])

cols = all_distributions_comparison.columns.tolist()
cols.insert(1, cols.pop(cols.index('family')))
all_distributions_comparison = all_distributions_comparison[cols]
# Grouped by response, ascending AICc within each response group.
all_distributions_comparison = all_distributions_comparison.sort_values(
    ['response', 'aicc'], ascending=[True, True]).reset_index(drop=True)
all_distributions_comparison.to_csv(
    '/Users/chenyjin/Documents/analysis/chap4_mito_nuclear/260513/true_genus_mito_plastid/filter_perc_id/plots/new/count_all_distributions_full_comparison.include_0.csv',
    index=False)

print(f"\nWrote merged summary-statistics table for all {len(all_distributions_comparison)} fitted models "
      f"(responses: {sorted(all_distributions_comparison['response'].unique())}; "
      f"families: {sorted(all_distributions_comparison['family'].unique())}; "
      f"{n_flagged} rows flagged is_reported_best_model, one per response) to "
      f"count_all_distributions_full_comparison.csv.")

# ---------------------------------------------------------------------------
# 7. Negative Binomial assumption-validity summary: linktest_eta_coef/eta_sq_pvalue/
#    mean_function_ok (mean-function/log-link test) and gof_pvalue/variance_function_ok
#    (variance-function test) are computed for every NB row directly in section 3's per-formula
#    loop above (and saved there in count_predictor_model_comparison.csv and the combined
#    count_all_distributions_full_comparison.csv for all formulas, not just the best one). This
#    just prints/exports the best-per-response row as a quick-reference summary, filtering the
#    already-computed columns rather than re-fitting anything.
# ---------------------------------------------------------------------------
nb_rows_all = count_comparison[count_comparison['family'] == 'negativebinomial']

nb_validity_rows = []
for response_name, ratio_col, genus_set in RESPONSE_CONFIGS:
    nb_candidates = nb_rows_all[nb_rows_all['response'] == response_name]
    if len(nb_candidates) == 0:
        print(f"\nNo converged Negative-Binomial model for {response_name} -- "
              f"can't test its assumptions.")
        continue
    nb_validity_rows.append(nb_candidates.loc[nb_candidates['aicc'].idxmin()])

nb_validity = pd.DataFrame(nb_validity_rows)
nb_validity.to_csv(
    '/Users/chenyjin/Documents/analysis/chap4_mito_nuclear/260513/true_genus_mito_plastid/filter_perc_id/plots/new/count_nb_assumption_validity.include_0.csv',
    index=False)

print("\n=== Negative Binomial assumption validity (best-AICc model per response; full table for "
      "every NB formula is in count_predictor_model_comparison.csv) ===")
print(nb_validity[['response', 'host_predictor', 'formula', 'n_obs', 'linktest_eta_coef',
                    'linktest_eta_sq_pvalue', 'mean_function_ok', 'dispersion_ratio', 'gof_pvalue',
                    'variance_function_ok']].to_string(index=False))

# ---------------------------------------------------------------------------
# 8. Compact log-likelihood/dispersion-ratio summary across every fitted model from section 3's
#    free-recombination search (both families, every host-identity/GTDB/denom/candidate-
#    predictor combination) -- lets log-likelihood and dispersion be scanned across the whole
#    search space without the full coefficient columns in count_predictor_model_comparison.csv.
# ---------------------------------------------------------------------------
loglik_dispersion_summary = count_comparison[
    ['response', 'family', 'host_predictor', 'formula', 'includes_gtdb',
     'includes_age', 'includes_feature', 'includes_project_name',
     'includes_genus_interaction', 'includes_feature_interaction', 'n_obs', 'df_model',
     'log_likelihood', 'dispersion_ratio', 'aic', 'bic']
].sort_values(['response', 'family', 'log_likelihood'], ascending=[True, True, False])
loglik_dispersion_summary.to_csv(
    '/Users/chenyjin/Documents/analysis/chap4_mito_nuclear/260513/true_genus_mito_plastid/filter_perc_id/plots/new/count_loglik_dispersion_summary.include_0.csv',
    index=False)

print(f"\nWrote log-likelihood/dispersion-ratio summary for all {len(loglik_dispersion_summary)} fitted "
      f"models to count_loglik_dispersion_summary.csv.")

# ---------------------------------------------------------------------------
# 9. Per-response Negative Binomial model with the lowest AICc (not raw log-likelihood, which is
#    monotonically non-decreasing as predictors are added to a nested MLE fit and would just pick
#    the most saturated formula tried for each response), keeping every
#    coefficient/p-value/standard-error column plus all assumption-check columns (VIF,
#    homoscedasticity, link test, goodness-of-fit). This is the same row already selected in
#    section 7's count_nb_assumption_validity.csv (built from the same nb_candidates.idxmin('aicc')
#    logic); this export just isolates it with log_likelihood alongside for reference.
# ---------------------------------------------------------------------------
nb_candidates_all = count_comparison[count_comparison['family'] == 'negativebinomial']

nb_best_aic_rows = []
for response_name, ratio_col, genus_set in RESPONSE_CONFIGS:
    candidates = nb_candidates_all[nb_candidates_all['response'] == response_name]
    if len(candidates) == 0:
        print(f"\nNo converged Negative-Binomial model for {response_name} -- "
              f"can't select a best-by-AICc model.")
        continue
    nb_best_aic_rows.append(candidates.loc[candidates['aicc'].idxmin()])

nb_best_aic = pd.DataFrame(nb_best_aic_rows)
nb_best_aic.to_csv(
    '/Users/chenyjin/Documents/analysis/chap4_mito_nuclear/260513/true_genus_mito_plastid/filter_perc_id/plots/new/count_nb_best_aic_full.include_0.csv',
    index=False)

print("\n=== Negative Binomial model with the lowest AICc per response (full coefficients, "
      "p-values, and assumption checks in count_nb_best_aic_full.csv) ===")
print(nb_best_aic[['response', 'host_predictor', 'formula', 'n_obs', 'df_model', 'aic', 'aicc',
                    'log_likelihood', 'dispersion_ratio', 'assumptions_ok', 'mean_function_ok',
                    'variance_function_ok']].to_string(index=False))

# ---------------------------------------------------------------------------
# 10. Residuals-vs-fitted diagnostic plot for each response's best-supported Negative Binomial
#     model (the same rows as nb_best_aic_rows above). Fitted nuclear read counts span several
#     orders of magnitude across samples (from single digits to thousands), so a linear x-axis
#     would crush nearly every point into one corner -- fitted values are plotted on a log scale.
#     Residuals themselves stay on their native (Pearson) scale on the y-axis: a random scatter
#     centered on zero with no trend or funnel shape across the fitted range is the visual
#     complement to the Breusch-Pagan/goodness-of-fit tests computed in section 3.
# ---------------------------------------------------------------------------
from matplotlib.ticker import ScalarFormatter, LogLocator, NullLocator

INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
AXIS_LINE = '#c3c2b7'
POINT_COLOR = '#2a78d6'

fig, axes = plt.subplots(1, len(nb_best_aic_rows), figsize=(5 * len(nb_best_aic_rows), 4.5))
if len(nb_best_aic_rows) == 1:
    axes = [axes]

for ax, best_row in zip(axes, nb_best_aic_rows):
    response_name = best_row['response']
    fit = count_fits[(response_name, 'negativebinomial', best_row['formula'])]
    fitted = fit.predict(which='mean')  # fit.fittedvalues is the linear predictor Xb (log
                                         # scale) for this discrete model, not the response-scale
                                         # mean -- see the nb_lr_gof call site above
    resid = fit.resid_pearson

    ax.scatter(fitted, resid, s=36, color=POINT_COLOR, alpha=0.85, edgecolor='none')
    ax.axhline(0, color=INK_MUTED, linewidth=1)
    ax.set_xscale('log')
    # Default log-locator scientific notation (e.g. "6x10^0") reads poorly for a range this
    # narrow (well under a decade for some responses) -- plain numbers are clearer here. Minor
    # ticks are suppressed entirely so a stray minor label can't appear alongside the major ones.
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5, 7)))
    major_formatter = ScalarFormatter()
    major_formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(major_formatter)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Fitted nuclear read count (log scale)', color=INK_SECONDARY)
    ax.set_ylabel('Pearson residual', color=INK_SECONDARY)
    ax.set_title(f"{response_name}\nAICc = {best_row['aicc']:.1f}, n = {int(best_row['n_obs'])}",
                 color=INK_PRIMARY, fontsize=10)
    ax.grid(True, which='major', color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(AXIS_LINE)
    ax.tick_params(colors=INK_MUTED)

fig.suptitle('Residuals vs. fitted -- best-AICc Negative Binomial model per response', color=INK_PRIMARY)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(
    '/Users/chenyjin/Documents/analysis/chap4_mito_nuclear/260513/true_genus_mito_plastid/filter_perc_id/plots/new/residuals_vs_fitted.pdf',
    dpi=150)
fig.savefig(
    '/Users/chenyjin/Documents/analysis/chap4_mito_nuclear/260513/true_genus_mito_plastid/filter_perc_id/plots/new/residuals_vs_fitted.png',
    dpi=150)
plt.close(fig)

print(f"\nWrote residuals-vs-fitted diagnostic plot for {len(nb_best_aic_rows)} responses to "
      f"residuals_vs_fitted.pdf/.png.")
