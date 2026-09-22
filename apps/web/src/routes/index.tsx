import { createFileRoute } from "@tanstack/react-router";
import {
	ArrowDown,
	ArrowUpRight,
	Asterisk,
	Braces,
	CircleDot,
	Play,
	Plus,
	X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

export const Route = createFileRoute("/")({ component: Home });

const repositoryUrl = "https://github.com/tnspacetime/system-one-plus";
const xUrl = "https://x.com/__tuan____";

const researchQuestions = [
	{
		index: "01",
		title: "Flag what may be missing",
		body: "Can the system recognize that the supplied action menu may be incomplete?",
	},
	{
		index: "02",
		title: "Surface what should be added",
		body: "When needed, can it propose the action that belongs in the decision?",
	},
];

const useCases = [
	{
		index: "01",
		label: "Agent tools",
		title: "The capability is absent.",
		body: "A coding agent has read, edit, and test. The environment needs a missing dependency or a human escalation. Do not force another useless tool call.",
	},
	{
		index: "02",
		label: "Infrastructure",
		title: "The runbook is wrong.",
		body: "Restart, scale, and clear cache cannot repair an external DNS failure. Flag the runbook before automation thrashes the system.",
	},
	{
		index: "03",
		label: "Fraud operations",
		title: "The unit of action changed.",
		body: "Approve, decline, and challenge operate on one payment. A coordinated attack may require campaign-level containment.",
	},
	{
		index: "04",
		label: "Workflow telemetry",
		title: "Every omission is a signal.",
		body: "Aggregate inadequate menus into a live map of where tools, policies, and operating procedures fail to cover reality.",
	},
];

const specificationRequirements = [
	{
		index: "01",
		label: "Detect",
		title: "Know when the menu may be incomplete.",
		body: "Flag when the supplied choices may omit a better action.",
	},
	{
		index: "02",
		label: "Surface",
		title: "Say what should be added.",
		body: "When needed, propose candidate actions outside the supplied menu.",
	},
];

function Home() {
	return (
		<main id="top" className="site-shell">
			<SiteHeader />
			<Hero />
			<ConsequenceSection />
			<CapacitySection />
			<DualitySection />
			<SpecificationSection />
			<ArchitectureSection />
			<UseCasesSection />
			<ResearchSection />
			<ClosingSection />
			<SiteFooter />
		</main>
	);
}

function SiteHeader() {
	return (
		<header className="site-header">
			<a className="wordmark" href="#top" aria-label="System One Plus home">
				<span>System One</span>
				<Plus aria-hidden="true" />
			</a>
			<nav className="site-nav" aria-label="Primary navigation">
				<a href="#thesis">Thesis</a>
				<a href="#specification">Specification</a>
				<a href="#architecture">Architecture</a>
				<a href="#applications">Cases</a>
				<a href="#research">Research</a>
			</nav>
			<div className="header-links">
				<a
					className="header-index"
					href={repositoryUrl}
					target="_blank"
					rel="noreferrer"
				>
					GitHub <ArrowUpRight aria-hidden="true" />
				</a>
				<a
					className="header-index"
					href={xUrl}
					target="_blank"
					rel="noreferrer"
				>
					X <ArrowUpRight aria-hidden="true" />
				</a>
			</div>
		</header>
	);
}

function Hero() {
	const launchFilmDialog = useRef<HTMLDialogElement>(null);
	const launchFilm = useRef<HTMLVideoElement>(null);

	function openLaunchFilm() {
		launchFilmDialog.current?.showModal();
		void launchFilm.current?.play();
	}

	function closeLaunchFilm() {
		launchFilm.current?.pause();
		launchFilmDialog.current?.close();
	}

	return (
		<section className="hero" id="thesis">
			<div className="hero-grid" aria-hidden="true" />
			<div className="hero-orbit orbit-one" aria-hidden="true" />
			<div className="hero-orbit orbit-two" aria-hidden="true" />

			<div className="hero-meta reveal reveal-one">
				<span>A research proposition</span>
				<span>Working paper 00</span>
			</div>

			<div className="hero-copy">
				<p className="eyebrow reveal reveal-two">
					Beyond closed-set machine intelligence
				</p>
				<h1 className="reveal reveal-three">
					The menu
					<br />
					is not <em>the world.</em>
				</h1>
				<div className="hero-deck reveal reveal-four">
					<p>
						A foundation model that can recognize a better action should not be
						forbidden from proposing it.
					</p>
					<a
						className="round-link"
						href="#asymmetry"
						aria-label="Read the argument"
					>
						<ArrowDown aria-hidden="true" />
					</a>
				</div>
			</div>

			<div className="hero-bottom">
				<button
					type="button"
					className="hero-film-card reveal reveal-five"
					onClick={openLaunchFilm}
					aria-haspopup="dialog"
				>
					<span className="hero-film-thumb" aria-hidden="true">
						<img
							src="/system-one-plus-launch-film-poster.png"
							alt=""
							width="1920"
							height="1080"
						/>
						<span className="hero-film-play">
							<Play aria-hidden="true" />
						</span>
					</span>
					<span className="hero-film-copy">
						<small>Launch film · 02:53</small>
						<strong>Watch System One+</strong>
						<span>
							The argument in motion <ArrowUpRight aria-hidden="true" />
						</span>
					</span>
				</button>

				<div className="hero-footnote reveal reveal-five">
					<Asterisk aria-hidden="true" />
					<p>
						System One+ is a proposal for decision systems that can challenge
						the boundaries of the question—not merely choose inside them.
					</p>
				</div>
			</div>

			<dialog
				ref={launchFilmDialog}
				className="launch-film-dialog"
				aria-labelledby="launch-film-title"
				onClose={() => launchFilm.current?.pause()}
				onClick={(event) => {
					if (event.target === event.currentTarget) closeLaunchFilm();
				}}
				onKeyDown={(event) => {
					if (event.key === "Escape") closeLaunchFilm();
				}}
			>
				<div className="launch-film-frame">
					<header>
						<div>
							<span>Launch film / 02:53</span>
							<h2 id="launch-film-title">System One+</h2>
						</div>
						<button
							type="button"
							onClick={closeLaunchFilm}
							aria-label="Close launch film"
						>
							<X aria-hidden="true" />
						</button>
					</header>
					<video
						ref={launchFilm}
						controls
						playsInline
						preload="metadata"
						poster="/system-one-plus-launch-film-poster.png"
					>
						<source src="/system-one-plus-launch-film.mp4" type="video/mp4" />
						<track
							kind="captions"
							src="/system-one-plus-launch-film.vtt"
							srcLang="en"
							label="English"
						/>
						Your browser does not support embedded video.
					</video>
				</div>
			</dialog>
		</section>
	);
}

function ConsequenceSection() {
	return (
		<section className="consequence section-pad" id="consequence">
			<SectionMarker number="01" label="The consequence" light />
			<div className="consequence-header">
				<div>
					<p className="eyebrow">Scores do not stay on the screen.</p>
					<h2>
						Software acts
						<br />
						<em>on them.</em>
					</h2>
				</div>
				<p>
					A decision system routes a payment, invokes a tool, restarts a
					service, changes access, or escalates a person. The output leaves the
					model and changes the world.
				</p>
			</div>

			<div
				className="consequence-chain"
				role="img"
				aria-label="A score becomes an action, and an action creates a consequence"
			>
				<div>
					<span>01 / output</span>
					<strong>Score</strong>
					<b>0.91</b>
				</div>
				<ArrowUpRight aria-hidden="true" />
				<div>
					<span>02 / execution</span>
					<strong>Action</strong>
					<b>Run</b>
				</div>
				<ArrowUpRight aria-hidden="true" />
				<div>
					<span>03 / reality</span>
					<strong>Consequence</strong>
					<b>Real</b>
				</div>
			</div>

			<div className="consequence-question">
				<p>So the goal is not the best score.</p>
				<h3>The goal is the best possible action.</h3>
				<div>
					<Asterisk aria-hidden="true" />
					<p>
						If you trust the evaluator enough to act on its judgment, can you
						guarantee the best action was already in your menu?
					</p>
				</div>
			</div>
		</section>
	);
}

function CapacitySection() {
	return (
		<section className="capacity section-pad" id="asymmetry">
			<SectionMarker number="02" label="The asymmetry" />
			<div className="capacity-headline">
				<h2>
					You paid for
					<br />a brain.
				</h2>
				<h2 className="outline-type">
					You built
					<br />a switch.
				</h2>
			</div>

			<div className="capacity-grid">
				<div
					className="capacity-map"
					role="img"
					aria-label="Foundation model capacity compared with a small option menu"
				>
					<div className="capacity-noise" aria-hidden="true" />
					<div className="capacity-window">
						<span>3 options</span>
					</div>
					<div className="capacity-map-label top">Latent world model</div>
					<div className="capacity-map-label bottom">
						Human-authored boundary
					</div>
				</div>
				<div className="capacity-copy">
					<p className="lead-copy">
						We stream billions of learned parameters through expensive hardware,
						then permit the result to select between a handful of strings
						written in advance.
					</p>
					<div
						className="ratio-lockup"
						role="img"
						aria-label="Billions of parameters compressed into three options"
					>
						<div>
							<strong>N</strong>
							<span>learned parameters</span>
						</div>
						<div className="ratio-arrow" aria-hidden="true">
							<span />
							<ArrowUpRight />
						</div>
						<div>
							<strong>K</strong>
							<span>supplied actions</span>
						</div>
					</div>
					<p className="body-copy">
						The model may contain useful operational knowledge that the
						interface never asks for. Closed-set ranking turns that knowledge
						into a passive filter: powerful enough to notice the omission,
						structurally unable to name it.
					</p>
				</div>
			</div>
		</section>
	);
}

function DualitySection() {
	return (
		<section className="duality section-pad">
			<SectionMarker number="03" label="The opening" />
			<div className="duality-intro">
				<h2>
					Ranking samples
					<br />
					the surface.
				</h2>
				<p>
					If a model assigns compatibility to arbitrary actions, the supplied
					menu is only a set of coordinates. The unexplored landscape still
					matters.
				</p>
			</div>

			<div
				className="equation"
				role="math"
				aria-label="Probability of an action given a state is proportional to the exponential utility of that state and action"
			>
				<span>P(c | x)</span>
				<i>∝</i>
				<span>exp</span>
				<b>(</b>
				<span className="equation-accent">u(x, c)</span>
				<b>)</b>
			</div>

			<DecisionField />

			<p className="duality-note">
				Scoring candidates does not automatically produce new ones. Action-space
				expansion is the next research problem.
			</p>
		</section>
	);
}

function DecisionField() {
	const [plusMode, setPlusMode] = useState(false);

	useEffect(() => {
		if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
			return;
		}

		const timer = window.setTimeout(
			() => setPlusMode((currentMode) => !currentMode),
			plusMode ? 5200 : 3600,
		);

		return () => window.clearTimeout(timer);
	}, [plusMode]);

	return (
		<div className={`decision-demo ${plusMode ? "is-plus" : ""}`}>
			<div className="decision-toolbar">
				<span className="mode-progress" aria-hidden="true" />
				<div>
					<span className="status-light" aria-hidden="true" />
					Decision field / illustrative
				</div>
				<fieldset className="mode-toggle">
					<legend className="sr-only">Decision field mode</legend>
					<button
						type="button"
						className={!plusMode ? "is-active" : ""}
						aria-pressed={!plusMode}
						onClick={() => setPlusMode(false)}
					>
						Closed set
					</button>
					<button
						type="button"
						className={plusMode ? "is-active" : ""}
						aria-pressed={plusMode}
						onClick={() => setPlusMode(true)}
					>
						System One+
					</button>
				</fieldset>
			</div>

			<div className="decision-state">
				<span>STATE / 04:18:09</span>
				<p>
					High-value transfer · new device · account recovery six hours ago ·
					customer currently abroad
				</p>
			</div>

			<div className="decision-landscape">
				<div className="contour contour-one" aria-hidden="true" />
				<div className="contour contour-two" aria-hidden="true" />
				<div className="contour contour-three" aria-hidden="true" />
				<div className="scan-line" aria-hidden="true" />

				<Candidate
					className="candidate-a"
					code="A"
					label="Decline transfer"
					score=".42"
				/>
				<Candidate
					className="candidate-b"
					code="B"
					label="Standard review"
					score=".58"
				/>
				<Candidate
					className="candidate-c"
					code="C"
					label="Approve transfer"
					score=".17"
				/>
				<Candidate
					className="candidate-d"
					code="D+"
					label="Secure recovery channel, then hold transfer"
					score=".91"
					proposed
				/>

				<div className="field-caption">
					{plusMode ? (
						<>
							<CircleDot aria-hidden="true" /> The evaluator recommends a
							candidate outside the supplied menu.
						</>
					) : (
						<>
							<Braces aria-hidden="true" /> The system may only compare A, B,
							and C.
						</>
					)}
				</div>
			</div>
		</div>
	);
}

function Candidate({
	className,
	code,
	label,
	score,
	proposed = false,
}: {
	className: string;
	code: string;
	label: string;
	score: string;
	proposed?: boolean;
}) {
	return (
		<div className={`candidate ${className}`}>
			<span className="candidate-dot" aria-hidden="true" />
			<div>
				<small>{proposed ? "proposed" : `option ${code}`}</small>
				<strong>{label}</strong>
			</div>
			<b>{score}</b>
		</div>
	);
}

function SpecificationSection() {
	return (
		<section className="specification section-pad" id="specification">
			<SectionMarker number="04" label="Minimal contract" />
			<div className="specification-header">
				<div>
					<p className="eyebrow">System One + two capabilities</p>
					<h2>That is the plus.</h2>
				</div>
				<p>
					Detect when the action menu may be incomplete. When needed, surface
					what should be added. Everything else is an implementation choice and
					a research question.
				</p>
			</div>

			<ol className="specification-list">
				{specificationRequirements.map((requirement) => (
					<li key={requirement.index}>
						<span>{requirement.index}</span>
						<p>{requirement.label}</p>
						<h3>{requirement.title}</h3>
						<p>{requirement.body}</p>
					</li>
				))}
			</ol>
		</section>
	);
}

function ArchitectureSection() {
	return (
		<section className="architecture section-pad" id="architecture">
			<SectionMarker number="05" label="Reference architecture" light />
			<div className="architecture-header">
				<p className="eyebrow">One trained decision system. Two paths.</p>
				<h2>
					Fast by default.
					<br />
					<span>Generative by exception.</span>
				</h2>
				<p>
					System One+ does not turn every decision into a conversation.
					Generation is a repair path, opened only when the menu may be
					inadequate and the caller requests a missing action.
				</p>
			</div>

			<div className="architecture-paths">
				<article className="architecture-path common-path">
					<div className="path-label">
						<span>01</span>
						<p>Common path</p>
					</div>
					<h3>The menu is adequate.</h3>
					<div
						className="path-flow"
						role="img"
						aria-label="Common inference path"
					>
						<span>State</span>
						<i />
						<span>Score</span>
						<i />
						<span>Check</span>
						<i />
						<span>Decide</span>
					</div>
					<p>
						Return typed scores and an omission signal. No proposal needs to be
						decoded.
					</p>
				</article>

				<article className="architecture-path exception-path">
					<div className="path-label">
						<span>02</span>
						<p>Exception path</p>
					</div>
					<h3>The menu may be incomplete.</h3>
					<div
						className="path-flow"
						role="img"
						aria-label="Exception inference path"
					>
						<span>Flag</span>
						<i />
						<span>Propose</span>
						<i />
						<span>Score</span>
						<i />
						<span>Expand</span>
					</div>
					<p>
						The flag can return immediately. If repair is requested, propose a
						missing action and score it on the same decision surface.
					</p>
				</article>
			</div>

			<div className="architecture-rule">
				<Asterisk aria-hidden="true" />
				<p>
					A flag is already useful. Generation should not be a tax on every
					request.
				</p>
			</div>
		</section>
	);
}

function UseCasesSection() {
	return (
		<section className="use-cases section-pad" id="applications">
			<SectionMarker number="06" label="Where it matters" />
			<div className="use-cases-header">
				<h2>
					Better action spaces,
					<br />
					<span>not just better choices.</span>
				</h2>
				<p>
					Illustrative production settings where a least-bad choice can be more
					dangerous than an explicit omission.
				</p>
			</div>

			<div className="use-case-grid">
				{useCases.map((useCase) => (
					<article key={useCase.index}>
						<div>
							<span>{useCase.index}</span>
							<p>{useCase.label}</p>
						</div>
						<h3>{useCase.title}</h3>
						<p>{useCase.body}</p>
						<ArrowUpRight aria-hidden="true" />
					</article>
				))}
			</div>
		</section>
	);
}

function ResearchSection() {
	return (
		<section className="research section-pad" id="research">
			<SectionMarker number="07" label="Open research" />
			<div className="research-header">
				<h2>
					The prototype exists.
					<br />
					The problem is now concrete.
				</h2>
				<p>
					<a
						className="inline-repository-link"
						href={repositoryUrl}
						target="_blank"
						rel="noreferrer"
					>
						This repository <ArrowUpRight aria-hidden="true" />
					</a>{" "}
					is an executable prototype of System One+: flag what may be missing and,
					when needed, surface what should be added.
				</p>
			</div>

			<div className="research-list">
				{researchQuestions.map((question) => (
					<article key={question.index}>
						<span>{question.index}</span>
						<h3>{question.title}</h3>
						<p>{question.body}</p>
						<ArrowUpRight aria-hidden="true" />
					</article>
				))}
			</div>
		</section>
	);
}

function ClosingSection() {
	return (
		<section className="closing">
			<div className="closing-grid" aria-hidden="true" />
			<p className="eyebrow">The next problem after System One</p>
			<h2>
				Work on
				<br />
				System One+.
			</h2>
			<p className="closing-accent">Challenge the missing layer.</p>
			<div className="closing-cta">
				<p>
					Read the proposal. Run the prototype. Test the assumptions. Design the
					evaluations that could prove—or break—the idea.
				</p>
				<div className="closing-links">
					<a href={repositoryUrl} target="_blank" rel="noreferrer">
						View the repository <ArrowUpRight aria-hidden="true" />
					</a>
					<a href={xUrl} target="_blank" rel="noreferrer">
						X <ArrowUpRight aria-hidden="true" />
					</a>
				</div>
			</div>
			<div className="prototype-warning">
				<span>Early research prototype / open architecture</span>
				<p>
					Only proceed if you enjoy sweating over architecture, testing hard
					assumptions, and wondering whether the best action was missing all
					along. The architecture is here to be trained, challenged, and
					improved.
				</p>
			</div>
		</section>
	);
}

function SiteFooter() {
	return (
		<footer className="site-footer">
			<div className="wordmark footer-mark">
				<span>System One</span>
				<Plus aria-hidden="true" />
			</div>
			<div>
				<span>Working proposition / 00</span>
			</div>
			<a href="#top">Back to top ↑</a>
		</footer>
	);
}

function SectionMarker({
	number,
	label,
	light = false,
}: {
	number: string;
	label: string;
	light?: boolean;
}) {
	return (
		<div className={`section-marker ${light ? "is-light" : ""}`}>
			<span>{number}</span>
			<div />
			<p>{label}</p>
		</div>
	);
}
