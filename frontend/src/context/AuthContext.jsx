import { createContext, useContext, useEffect, useState } from 'react';
import {
    getMe,
    isAuthenticated,
    login as apiLogin,
    logout as apiLogout,
    signup as apiSignup,
} from '../api/client';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        let cancelled = false;
        (async () => {
            if (isAuthenticated()) {
                try {
                    const me = await getMe();
                    if (!cancelled) setUser(me);
                } catch {
                    if (!cancelled) setUser(null);
                }
            }
            if (!cancelled) setLoading(false);
        })();
        return () => {
            cancelled = true;
        };
    }, []);

    useEffect(() => {
        function onLogout() {
            setUser(null);
        }
        window.addEventListener('auth:logout', onLogout);
        return () => window.removeEventListener('auth:logout', onLogout);
    }, []);

    async function login(username, password) {
        const me = await apiLogin(username, password);
        setUser(me);
        return me;
    }

    async function signup(username, email, password) {
        const me = await apiSignup(username, email, password);
        setUser(me);
        return me;
    }

    function logout() {
        apiLogout();
        setUser(null);
    }

    return (
        <AuthContext.Provider value={{ user, loading, login, signup, logout }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error('useAuth must be used within an AuthProvider');
    return ctx;
}
