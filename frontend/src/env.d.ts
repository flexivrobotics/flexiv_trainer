declare module "react" {
    export type SetStateAction<S> = S | ((prevState: S) => S);
    export type Dispatch<A> = (value: A) => void;
    export function useState<S>(initialState: S | (() => S)): [S, Dispatch<SetStateAction<S>>];
    export function useEffect(effect: () => void | (() => void), deps?: readonly unknown[]): void;
    export function useMemo<T>(factory: () => T, deps: readonly unknown[]): T;
    export function useCallback<T extends (...args: any[]) => any>(callback: T, deps: readonly unknown[]): T;
    export function startTransition(scope: () => void): void;
    export const StrictMode: any;

    const React: any;
    export default React;
}

declare module "react-dom/client" {
    export function createRoot(container: Element | DocumentFragment): { render(node: any): void };
}

declare module "react-router-dom" {
    export const BrowserRouter: any;
    export const Link: any;
    export const NavLink: any;
    export const Route: any;
    export const Routes: any;
}

declare module "react/jsx-runtime" {
    export const Fragment: any;
    export const jsx: any;
    export const jsxs: any;
}

declare namespace JSX {
    interface IntrinsicElements {
        [elemName: string]: any;
    }
}

interface ImportMetaEnv {
    readonly VITE_API_BASE?: string;
}

interface ImportMeta {
    readonly env: ImportMetaEnv;
}
