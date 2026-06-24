import { useState } from 'react';
import { useAuth } from '../context/AuthContext';

export default function LoginPage({ onNavigateSignup }) {
    const { login } = useAuth();
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [submitting, setSubmitting] = useState(false);

    async function handleSubmit(e) {
        e.preventDefault();
        setError('');
        setSubmitting(true);
        try {
            await login(username, password);
        } catch (err) {
            setError(err?.message || 'Login failed');
        } finally {
            setSubmitting(false);
        }
    }

    return (
        <div className="auth-page">
            <form className="auth-card" onSubmit={handleSubmit}>
                <div className="auth-brand">
                    <span className="wps-logo-w">W</span>
                    <span className="auth-brand-name">DAL Word</span>
                </div>
                <h1 className="auth-title">Log in</h1>
                {error && <div className="auth-error">{error}</div>}
                <label className="auth-label">
                    Username
                    <input
                        className="auth-input"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        autoFocus
                        required
                    />
                </label>
                <label className="auth-label">
                    Password
                    <input
                        className="auth-input"
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                    />
                </label>
                <button className="auth-submit" type="submit" disabled={submitting}>
                    {submitting ? 'Logging in…' : 'Log in'}
                </button>
                <p className="auth-switch">
                    Don't have an account?{' '}
                    <button type="button" className="auth-link" onClick={onNavigateSignup}>
                        Sign up
                    </button>
                </p>
            </form>
        </div>
    );
}
