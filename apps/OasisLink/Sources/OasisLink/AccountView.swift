import SwiftUI

/// Sign in or create an account; the same form on the Mac and the phone.
public struct AccountView: View {
    @ObservedObject var store: AccountStore
    @State private var email = ""
    @State private var password = ""
    @State private var creating = false

    public init(store: AccountStore) {
        self.store = store
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let session = store.session {
                HStack(spacing: 12) {
                    Image(systemName: "person.crop.circle.fill").font(.title).foregroundStyle(.green)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(session.email).font(.headline)
                        Text("Signed in").font(.subheadline).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Sign out") { store.signOut() }
                }
            } else if !store.isConfigured {
                Label("Accounts are not set up in this build (supabase.json is empty).",
                      systemImage: "person.crop.circle.badge.questionmark")
                    .foregroundStyle(.secondary)
            } else {
                Text(creating ? "Create an account" : "Sign in").font(.headline)
                TextField("Email", text: $email)
                    .textContentType(.emailAddress)
                    #if os(iOS)
                    .keyboardType(.emailAddress)
                    .textInputAutocapitalization(.never)
                    #endif
                    .autocorrectionDisabled()
                    .textFieldStyle(.roundedBorder)
                SecureField("Password", text: $password)
                    .textContentType(creating ? .newPassword : .password)
                    .textFieldStyle(.roundedBorder)
                if let error = store.error {
                    Text(error).font(.footnote).foregroundStyle(.red)
                }
                if store.awaitingConfirmation {
                    Text("Check your email to confirm the account, then sign in.")
                        .font(.footnote).foregroundStyle(.green)
                }
                HStack {
                    Button {
                        Task {
                            if creating { await store.signUp(email: email, password: password) }
                            else { await store.signIn(email: email, password: password) }
                        }
                    } label: {
                        if store.busy { ProgressView().controlSize(.small) } else { Text(creating ? "Create" : "Sign in") }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(store.busy || email.isEmpty || password.count < 6)
                    Button(creating ? "I have an account" : "Create an account") { creating.toggle() }
                        .buttonStyle(.plain)
                        .foregroundStyle(.blue)
                }
            }
        }
    }
}
