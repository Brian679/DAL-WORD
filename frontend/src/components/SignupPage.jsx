import { useState } from 'react';
import { useAuth } from '../context/AuthContext';

export default function SignupPage({ onNavigateLogin }) {
    const { signup } = useAuth();
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [submitting, setSubmitting] = useState(false);

    async function handleSubmit(e) {
        e.preventDefault();
        setError('');
        setSubmitting(true);
        try {
            await signup(username, email, password);
        } catch (err) {
            setError(err?.message || 'Signup failed');
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
                <h1 className="auth-title">Create your account</h1>
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
                    Email
                    <input
                        className="auth-input"
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
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
                    {submitting ? 'Creating account…' : 'Sign up'}
                </button>
                <p className="auth-switch">
                    Already have an account?{' '}
                    <button type="button" className="auth-link" onClick={onNavigateLogin}>
                        Log in
                    </button>
                </p>
            </form>
        </div>
    );
}
