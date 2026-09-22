import { createRootRoute, HeadContent, Scripts } from "@tanstack/react-router";

import { seo } from "../lib/seo";
import appCss from "../styles.css?url";

const SITE_URL = (
	(import.meta.env.VITE_SITE_URL as string | undefined) ??
	"https://system-one-plus.tnspacetime.com"
).replace(/\/$/, "");
const SITE_TITLE = "System One+ — The Menu Is Not the World";
const SITE_DESCRIPTION =
	"An open research specification for fast decision systems that can detect an incomplete action menu and propose what is missing.";

export const Route = createRootRoute({
	head: () => ({
		meta: [
			{
				charSet: "utf-8",
			},
			{
				name: "viewport",
				content: "width=device-width, initial-scale=1",
			},
			{
				name: "theme-color",
				content: "#faf8ee",
			},
			{
				name: "author",
				content: "tnspacetime (@__tuan____)",
			},
			{
				name: "twitter:creator",
				content: "@__tuan____",
			},
			{
				name: "robots",
				content: "index, follow",
			},
			...seo({
				title: SITE_TITLE,
				description: SITE_DESCRIPTION,
				keywords:
					"System One Plus, decision systems, foundation models, machine intelligence, closed-set ranking, open action space, AI research",
				url: SITE_URL ? "/" : undefined,
				siteUrl: SITE_URL,
			}),
		],
		links: [
			{
				rel: "stylesheet",
				href: appCss,
			},
			{
				rel: "author",
				href: "https://x.com/__tuan____",
			},
			...(SITE_URL
				? [
						{
							rel: "canonical",
							href: SITE_URL,
						},
					]
				: []),
			{
				rel: "icon",
				href: "/favicon.svg?v=1",
				type: "image/svg+xml",
			},
			{
				rel: "icon",
				type: "image/png",
				sizes: "32x32",
				href: "/favicon-32x32.png?v=1",
			},
			{
				rel: "icon",
				type: "image/png",
				sizes: "16x16",
				href: "/favicon-16x16.png?v=1",
			},
			{
				rel: "shortcut icon",
				href: "/favicon.ico?v=1",
			},
			{
				rel: "apple-touch-icon",
				sizes: "180x180",
				href: "/apple-touch-icon.png?v=1",
			},
			{
				rel: "manifest",
				href: "/site.webmanifest",
			},
		],
	}),
	shellComponent: RootDocument,
});

function RootDocument({ children }: { children: React.ReactNode }) {
	return (
		<html lang="en">
			<head>
				<HeadContent />
			</head>
			<body>
				{children}
				<Scripts />
			</body>
		</html>
	);
}
