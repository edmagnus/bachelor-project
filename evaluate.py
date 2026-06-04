import math
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import kv

import Lossfct_ghd as Lossfct
import model_generator
from GHD_simulator import sample_ghd, gh1_pdf, ghd_marginal_params

# Function to resolve the model file path, checking both the provided path and the saved_models directory.
def resolve_saved_model_path(model_file, saved_models_dir="saved_models"):
    model_path=Path(model_file)
    if not model_path.exists():
        model_path=Path(saved_models_dir)/model_file
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model file: {model_file}")
    return model_path

# Function to resolve the metadata file path, checking both the provided path and the same directory as the model file.
def resolve_metadata_path(model_path, metadata_file=None):
    if metadata_file is not None:
        metadata_path=Path(metadata_file)
        if not metadata_path.exists():
            metadata_path=model_path.parent/metadata_file
    else:
        metadata_path=model_path.with_name(model_path.stem+"_metadata.npz")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find metadata file: {metadata_path}. "
            "main_ghd.py saves this file next to the .pth model."
        )
    return metadata_path

# Function to load the model from the .pth file and return the model and its kwargs.
def load_model(model_path, device="cpu"):
    checkpoint=torch.load(model_path,map_location=device,weights_only=False)
    if isinstance(checkpoint,(list,tuple)) and len(checkpoint)==2:
        kwargs,state_dict=checkpoint
    elif isinstance(checkpoint,dict) and "kwargs" in checkpoint and "state_dict" in checkpoint:
        kwargs,state_dict=checkpoint["kwargs"],checkpoint["state_dict"]
    else:
        raise ValueError("Unsupported model checkpoint format.")
    model=model_generator.MLP(**kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model,kwargs

# Used to sample n samples from the input model, given seed for reproducibility
def sample_model(model,n,input_dim,seed,device="cpu"):
    generator=torch.Generator(device=device)
    generator.manual_seed(seed)
    with torch.no_grad():
        U=torch.randn((n,input_dim),generator=generator,dtype=torch.float32,device=device)
        samples=model(U).detach().cpu()
    return samples

# Function to create the characteristic function based on the GHD parameters, using Lossfct's implementation.
def make_charfct(params,device="cpu"):
    lam=float(params["lam"])
    alpha=float(params["alpha"])
    beta=torch.tensor(params["beta"],dtype=torch.float32).to(device)
    delta=float(params["delta"])
    mu=torch.tensor(params["mu"],dtype=torch.float32).to(device)
    Lambda=torch.tensor(params["Lambda"],dtype=torch.float32).to(device)
    def charfct(W):
        return Lossfct.ghd_char_fct_vec(W,lam,alpha,beta,delta,mu,Lambda,device)
    return charfct

# Function to estimate loss plus constant using Brück's Lossfct implementation, given samples Y, kernel samples W, the kernel matrix function, and the characteristic function.
# (Uses Brück's implementation in Lossfct)
def loss_plus_constant(Y, W, kernelmatrix, charfct, device="cpu"):
    Y = Y.detach().to(device)
    W = W.detach().to(device)

    with torch.no_grad():
        constant_estimate = Lossfct.estimate_constant_vec(W, charfct)
        loss_estimate = (
            Lossfct.lossfunction_vec(Y, kernelmatrix, charfct, W, device)
            + constant_estimate
        )

    return loss_estimate, constant_estimate

# Function to compute the MMD distance between two samples, using Brück's Lossfct implementation of the Gaussian kernel and the bandwidth.
def mmd_sample_to_sample(x, y, bandwidth, device="cpu"):
    x = x.detach().to(device)
    y = y.detach().to(device)

    with torch.no_grad():
        dist_to_true_sam = torch.tensor(0.0, dtype=torch.float32, device=device)

        for b in bandwidth:
            b_lossfct = 1.0 / float(b)

            dist_to_true_sam += (
                Lossfct.MMD_equal_case(x, device, b_lossfct)
                - Lossfct.MMD_mixed_case(x, y, device, b_lossfct)
                + Lossfct.MMD_equal_case(y, device, b_lossfct)
            )

        dist_to_true_sam = dist_to_true_sam / len(bandwidth)

    return dist_to_true_sam

# Function to compute the theoretical quantile for a given probability and component j, using the marginal parameters from Lemma A.1 and numerical integration of the PDF.
def theoretical_ghd_quantile(prob,lam,alpha,beta,delta,mu,Lambda,j):
    params_j=ghd_marginal_params(j,lam,alpha,beta,delta,mu,Lambda)

    def pdf(x):
        return float(gh1_pdf(x,**params_j))

    def cdf(x):
        val,err=quad(pdf,-np.inf,x,epsabs=1e-8,epsrel=1e-8,limit=200)
        return val

    center=params_j["mu"]
    lo=center-1.0
    hi=center+1.0
    while cdf(lo)>prob:
        lo=center-2.0*(center-lo)
    while cdf(hi)<prob:
        hi=center+2.0*(hi-center)
    return brentq(lambda x: cdf(x)-prob,lo,hi,xtol=1e-7,rtol=1e-7,maxiter=100)

# Function to create the loss-over-epoch graph. Takes the loss history as input, as well as paths for saving the plot.
def plot_loss(loss_history, plots_dir, stem):
    epsilon = 1e-5

    raw_loss = np.asarray(loss_history, dtype=float)
    shifted_loss = raw_loss - np.min(raw_loss) + epsilon
    shifted_loss_every_40 = shifted_loss[::40]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Comment out one of the two lines below to choose between plotting every epoch or every 40 epochs.
    # (Plotting every 40 epochs was done with 6000 epochs, plotting all was done for 200 epochs)

    # ax.plot(list(range(0, len(shifted_loss_every_40)*40, 40)), shifted_loss_every_40, marker="o", markersize=4, label="Shifted loss (every 40 epochs)")
    ax.plot(list(range(len(shifted_loss))), shifted_loss)

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Shifted loss (log scale)")
    ax.set_title("(Shifted) Loss over Epochs")

    ax.minorticks_on()
    ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.5, alpha=0.20)

    fig.tight_layout()
    path = plots_dir / f"{stem}_loss_over_epoch.pdf"
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path

# Function to plot the empirical vs theoretical marginal densities for each component.
def plot_quantiles(model_samples,exact_samples,theoretical_quantiles,probs,plots_dir,stem):
    model_np=model_samples.detach().cpu().numpy()
    exact_np=exact_samples.detach().cpu().numpy()
    dim=model_np.shape[1]
    ncols=min(2,dim)
    nrows=int(np.ceil(dim/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(7*ncols,4.5*nrows),squeeze=False)
    for j in range(dim):
        ax=axes[j//ncols][j%ncols]
        q_model=np.quantile(model_np[:,j],probs)
        q_exact=np.quantile(exact_np[:,j],probs)
        q_theory=theoretical_quantiles[j]
        ax.plot(probs,q_model,marker="o",label="Model")
        ax.plot(probs,q_exact,marker="o",label="Exact simulator")
        ax.plot(probs,q_theory,marker="o",label="Theoretical")
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Value")
        ax.set_title(f"Component {j+1}")
        ax.legend()
    for j in range(dim,nrows*ncols):
        axes[j//ncols][j%ncols].axis("off")
    fig.tight_layout()
    path=plots_dir/f"{stem}_quantile_comparison.pdf"
    fig.savefig(path,format="pdf")
    plt.close(fig)
    return path

# Function to calculate theoretical covariance matrix of the GHD (based on A.8, with an apparent error in the original formula corrected here)
def theoretical_covariance(lam, alpha, beta, delta, mu, Lambda):
    gamma = np.sqrt(alpha**2 - beta @ Lambda @ beta)
    zeta = delta * gamma
    R_lam = kv(lam + 1, zeta) / kv(lam, zeta)

    S_lam = (kv(lam + 2, zeta) * kv(lam, zeta) - kv(lam + 1, zeta)**2) \
            / kv(lam, zeta)**2
    theoretical_cov = ((delta / gamma) * R_lam) * Lambda \
                + ((delta**2 / gamma**2) * S_lam) * np.outer(Lambda @ beta, Lambda @ beta)
    
    return theoretical_cov

# Main evaluation function that combines all the steps: loading the model, sampling, computing losses, MMD, and plotting.
def evaluate(model_file,metadata_file=None,seed=2026,loss_sample_size=20000,mmd_sample_size=10000,plot_sample_size=50000,plots_dir="Plots",saved_models_dir="saved_models"):

    # Set seeds for reproducibility in evaluation.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device="cpu"
    model_path=resolve_saved_model_path(model_file,saved_models_dir)
    metadata_path=resolve_metadata_path(model_path,metadata_file)
    plots_dir=Path(plots_dir)
    plots_dir.mkdir(parents=True,exist_ok=True)

    metadata=np.load(metadata_path,allow_pickle=True)
    params={key: metadata[key] for key in metadata.files}
    lam=float(params["lam"])
    alpha=float(params["alpha"])
    beta=np.asarray(params["beta"],dtype=float)
    delta=float(params["delta"])
    mu=np.asarray(params["mu"],dtype=float)
    Lambda=np.asarray(params["Lambda"],dtype=float)
    bandwidth=[float(x) for x in np.asarray(params["bandwidth"],dtype=float)]
    dim=int(params["dim"])
    k=int(params["k"])
    input_dim=k*dim
    loss_history=np.asarray(params["eval_loss_history"],dtype=float)

    model,kwargs=load_model(model_path,device)
    charfct=make_charfct({"lam":lam,"alpha":alpha,"beta":beta,"delta":delta,"mu":mu,"Lambda":Lambda},device)


    # Approximate loss of final model, using Brück's Lossfct implementation.
    def kernelmatrix_cpu(X):
        return Lossfct.gaussian_kernelmatrix(X,device,bandwidth)

    model_loss_samples = sample_model(model, loss_sample_size, input_dim, seed, device)
    W = Lossfct.kernelsample_gaussian(loss_sample_size,device,dim,bandwidth)

    approx_loss, constant_estimate = loss_plus_constant(model_loss_samples,W,kernelmatrix_cpu,charfct,device)

    # Approximate loss of an exact GHD sample, using the same W and same estimated constant.
    rng = np.random.default_rng(seed + 1)
    exact_loss_np = sample_ghd(lam, alpha, beta, delta, mu, Lambda, n=loss_sample_size, rng=rng)
    exact_loss_samples = torch.tensor(exact_loss_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        exact_loss = (Lossfct.lossfunction_vec(exact_loss_samples,kernelmatrix_cpu,charfct,W,device) + constant_estimate)

    # MMD comparison between model sample and exact simulator sample
    model_mmd_samples = sample_model(model, mmd_sample_size, input_dim, seed + 2, device)
    exact_mmd_np = sample_ghd(
        lam,
        alpha,
        beta,
        delta,
        mu,
        Lambda,
        n=mmd_sample_size,
        rng=np.random.default_rng(seed + 3),
    )
    exact_mmd_samples = torch.tensor(exact_mmd_np, dtype=torch.float32).to(device)

    dist_to_true_sam = mmd_sample_to_sample(model_mmd_samples,exact_mmd_samples,bandwidth,device)

    # Paths for plots, using the same stem as the model file.
    stem=model_path.stem
    loss_plot_path=plot_loss(loss_history,plots_dir,stem)

    # Quantile plot
    probs=np.array([0.01,0.05,0.1,0.3,0.5,0.7,0.9,0.95,0.99])
    model_plot_samples=sample_model(model,plot_sample_size,input_dim,seed+4,device)
    exact_plot_np=sample_ghd(lam,alpha,beta,delta,mu,Lambda,n=plot_sample_size,rng=np.random.default_rng(seed+5))
    exact_plot_samples=torch.tensor(exact_plot_np,dtype=torch.float32)
    theoretical_quantiles=[]
    for j in range(dim):
        theoretical_quantiles.append(np.array([theoretical_ghd_quantile(p,lam,alpha,beta,delta,mu,Lambda,j) for p in probs]))

    quantile_plot_path=plot_quantiles(model_plot_samples,exact_plot_samples,theoretical_quantiles,probs,plots_dir,stem)

    results={
        "approx_loss_model": float(approx_loss.detach().cpu()),
        "approx_loss_exact_sample": float(exact_loss.detach().cpu()),
        "mmd_model_vs_exact": float(dist_to_true_sam.detach().cpu()),
        "constant_estimate": float(constant_estimate.detach().cpu()),
        "loss_plot_path": str(loss_plot_path),
        "quantile_plot_path": str(quantile_plot_path),
        "probabilities": probs,
        "theoretical_quantiles": theoretical_quantiles,
    }

    # Compare theoretical covariance matrix of the GHD with the empirical covariance matrix of the model samples, and print both.
    theoretical_cov = theoretical_covariance(lam, alpha, beta, delta, mu, Lambda)

    model_samples_for_cov = sample_model(
        model, 100_000, input_dim, seed + 6, device
    ).detach().cpu().numpy()

    exact_cov_np = sample_ghd(
        lam, alpha, beta, delta, mu, Lambda,
        n=100_000,
        rng=np.random.default_rng(seed + 7),
    )

    model_cov = np.cov(model_samples_for_cov, rowvar=False)
    exact_cov = np.cov(exact_cov_np, rowvar=False)

    print("Theoretical covariance:")
    print(theoretical_cov)

    print("Exact simulator covariance:")
    print(exact_cov)

    print("Model covariance:")
    print(model_cov)

    print("Exact - theoretical:")
    print(exact_cov - theoretical_cov)

    print("Model - theoretical:")
    print(model_cov - theoretical_cov)

    print("The approximate loss of the final model is equal to",results["approx_loss_model"],". It should be close to 0.")
    print("The approximate loss of the exact sample is equal to",results["approx_loss_exact_sample"],". It should be close to 0.")
    print("The estimated MMD distance of a sample from the model to an exact sample is",results["mmd_model_vs_exact"],". It should be close to 0.")
    print("Saved loss plot to",results["loss_plot_path"])
    print("Saved quantile plot to",results["quantile_plot_path"])

    return results


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Evaluate a GHD generator trained by main_ghd.py")
    parser.add_argument("model_file",help="Model filename inside saved_models/ or full path to .pth file")
    parser.add_argument("--metadata_file",default=None,help="Optional corresponding *_metadata.npz file")
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--loss_sample_size",type=int,default=20000)
    parser.add_argument("--mmd_sample_size",type=int,default=10000)
    parser.add_argument("--plot_sample_size",type=int,default=50000)
    parser.add_argument("--plots_dir",default="Plots")
    parser.add_argument("--saved_models_dir",default="saved_models")
    args=parser.parse_args()
    evaluate(**vars(args))

    # Calculate covariance matrix of the model samples and print it, to compare with the theoretical covariance matrix of the GHD.
    # (This is not part of the main evaluation function, but can be useful for further analysis.)
    # model_samples_np=model_samples.detach().cpu().numpy()
